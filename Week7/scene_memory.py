##### scene_memory.py
# Scene scrapbook + motion logger for robot runs.
# Logs trajectory, events, grids, meta, blocks for analysis.
# This version adds:
# - Rate limiting (max_hz) and motion gating (min_dpos/min_dang)
# - Safe/truncated handling of big "extra" payloads
# - Gzip compression for heavy CSVs in scene_memory/_raw/<run_tag>/

import os
import csv
import json
import time
import gzip
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np

from grid_utils import extract_scene, scene_to_grid, grid_to_positions


# ---- small helpers ----------------------------------------------------------

def _now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _to_list(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_list(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_list(v) for v in x]
    return x

def _nearest_cell(model, data, world_xyz: np.ndarray, grid_shape: Tuple[int, int]) -> Tuple[int, int, float]:
    rows, cols = grid_shape
    best_row, best_col, best_d = 0, 0, float("inf")
    for r in range(rows):
        for c in range(cols):
            p = grid_to_positions(model, data, r, c)
            d = float(np.linalg.norm(p - world_xyz))
            if d < best_d:
                best_row, best_col, best_d = r, c, d
    return best_row, best_col, best_d


# ---- compact records we store -----------------------------------------------

@dataclass
class BlockRecord:
    # Block position, grid mapping, and color
    name: str
    x: float
    y: float
    z: float
    row: int
    col: int
    cell_dist: float
    color: Optional[Any] = None

@dataclass
class SnapshotMeta:
    # Metadata about a snapshot run
    task_path: Optional[str]
    split: Optional[str]
    rule: Any
    rule_str: str
    plate_angle: float
    time_tag: str
    grid_shape: Tuple[int, int]


# ---- main API ---------------------------------------------------------------

class SceneMemory:
    """
    Handles saving robot run traces, events, and snapshots.
    Heavy artifacts (trajectory.csv, blocks.csv) go under: scene_memory/_raw/<run_tag>/
    """

    def __init__(
        self,
        root_dir: str = "scene_memory",
        heavy_subdir: str = "_raw",
        write_trajectory: bool = True,
        write_blocks: bool = True,
        # logging controls
        max_hz: float = 20.0,        # cap trajectory logging rate (Hz). 0 = no cap
        min_dpos: float = 1e-3,      # min EE translation to log (meters)
        min_dang: float = 1e-2,      # min EE rotation angle to log (radians) ~0.57°
        max_extra_chars: int = 2048, # truncate "extra" payloads to this many chars
        gzip_heavy: bool = True      # gzip heavy CSVs
    ):
        self.root_dir = root_dir
        self.heavy_subdir = heavy_subdir
        self.write_trajectory = write_trajectory
        self.write_blocks = write_blocks

        self.max_hz = float(max_hz)
        self.min_dpos = float(min_dpos)
        self.min_dang = float(min_dang)
        self.max_extra_chars = int(max_extra_chars)
        self.gzip_heavy = bool(gzip_heavy)

        _ensure_dir(self.root_dir)
        _ensure_dir(os.path.join(self.root_dir, self.heavy_subdir))

        self._reset_buffers()
        self._active_run_dir: Optional[str] = None
        self._heavy_run_dir: Optional[str] = None
        self._time0: Optional[float] = None

        # last-sample cache for motion gating
        self._last_log_t: float = -1e9
        self._last_log_pos: Optional[np.ndarray] = None
        self._last_log_quat: Optional[np.ndarray] = None

    # --------------- internals ----------------

    def _heavy_path(self, filename: str) -> str:
        assert self._heavy_run_dir is not None
        return os.path.join(self._heavy_run_dir, filename)

    def _quat_delta_angle(self, q_prev: np.ndarray, q_now: np.ndarray) -> float:
        # unit quaternions → angle between them
        d = abs(float(np.dot(q_prev, q_now)))
        d = min(1.0, max(-1.0, d))
        return 2.0 * float(np.arccos(d))

    def _should_log(self, t_now: float, pos: np.ndarray, quat: np.ndarray) -> bool:
        # rate limit
        if self.max_hz > 0:
            min_dt = 1.0 / self.max_hz
            if t_now - self._last_log_t < min_dt:
                return False
        # first sample always logs
        if self._last_log_pos is None or self._last_log_quat is None:
            return True
        # motion gating
        dpos = float(np.linalg.norm(pos - self._last_log_pos))
        dang = self._quat_delta_angle(self._last_log_quat, quat)
        return (dpos >= self.min_dpos) or (dang >= self.min_dang)

    def _sanitize_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        clean = {}
        for k, v in extra.items():
            try:
                # try to keep structure but cap size
                val = _to_list(v)
                txt = json.dumps(val, ensure_ascii=False)
            except Exception:
                txt = str(v)
            if len(txt) > self.max_extra_chars:
                txt = txt[: self.max_extra_chars] + "…"
            clean[k] = txt
        return clean

    # --------------- public API ----------------

    def trace_start(self, run_tag: Optional[str] = None) -> str:
        # Begin a new trace run and create folder
        if run_tag is None:
            run_tag = _now_tag()
        out_dir = os.path.join(self.root_dir, run_tag)
        _ensure_dir(out_dir)

        # Heavy files go under scene_memory/_raw/<run_tag>/
        heavy_dir = os.path.join(self.root_dir, self.heavy_subdir, run_tag)
        _ensure_dir(heavy_dir)

        self._active_run_dir = out_dir
        self._heavy_run_dir = heavy_dir
        self._time0 = time.time()
        self._reset_buffers()

        self._last_log_t = -1e9
        self._last_log_pos = None
        self._last_log_quat = None

        return out_dir

    def trace_step(self, model, data, phase: str = "", extra: Optional[Dict[str, Any]] = None) -> None:
        # Record one (possibly-throttled) timestep: pose, grip, plate angle
        if self._active_run_dir is None or self._time0 is None:
            return

        xpos = data.body("panda_hand").xpos.copy()
        xquat = data.body("panda_hand").xquat.copy()

        g1 = float(data.joint("panda_finger_joint1").qpos[0])
        g2 = float(data.joint("panda_finger_joint2").qpos[0])
        grip_aperture = g1 + g2

        try:
            plate_angle = float(data.joint("plate_rotation").qpos[0])
        except Exception:
            plate_angle = float("nan")

        t = time.time() - self._time0

        if not self._should_log(t, xpos, xquat):
            return

        self._last_log_t = t
        self._last_log_pos = xpos
        self._last_log_quat = xquat

        row = {
            "t": t,
            "x": float(xpos[0]), "y": float(xpos[1]), "z": float(xpos[2]),
            "qx": float(xquat[0]), "qy": float(xquat[1]), "qz": float(xquat[2]), "qw": float(xquat[3]),
            "plate_angle": plate_angle,
            "grip_aperture": grip_aperture,
            "phase": phase or ""
        }

        if extra:
            row.update(self._sanitize_extra(extra))

        self._traj.append(row)

    def log_event(self, model, data, *, kind: str, label: Optional[str] = None,
                  grid_shape: Optional[Tuple[int, int]] = None, note: Optional[str] = None) -> None:
        # Log event (pick/place) at current gripper pose
        if self._active_run_dir is None or self._time0 is None:
            return
        xpos = data.body("panda_hand").xpos.copy()
        xquat = data.body("panda_hand").xquat.copy()
        try:
            plate_angle = float(data.joint("plate_rotation").qpos[0])
        except Exception:
            plate_angle = float("nan")
        row = {
            "t": time.time() - self._time0,
            "kind": kind,
            "label": label or "",
            "x": float(xpos[0]), "y": float(xpos[1]), "z": float(xpos[2]),
            "qx": float(xquat[0]), "qy": float(xquat[1]), "qz": float(xquat[2]), "qw": float(xquat[3]),
            "plate_angle": plate_angle,
            "note": note or ""
        }
        if grid_shape is not None:
            r, c, d = _nearest_cell(model, data, xpos, grid_shape)
            row.update({"row": int(r), "col": int(c), "cell_dist": float(d)})
        self._events.append(row)

    def save_abort(self, model, data, *, input_grid: Optional[np.ndarray] = None,
                   grid_shape: Optional[Tuple[int, int]] = None, reason: str = "aborted") -> Optional[str]:
        # Save aborted run data and reason
        if self._active_run_dir is None:
            return None
        self._flush_trace_files(grid_shape=grid_shape)
        with open(os.path.join(self._active_run_dir, "abort.json"), "w", encoding="utf-8") as f:
            json.dump({"reason": reason, "time_tag": os.path.basename(self._active_run_dir)}, f, indent=2)
        out = self._active_run_dir
        self._active_run_dir = None
        self._heavy_run_dir = None
        return out

    def save_success(self, model, data, *, input_grid: np.ndarray, expected_output: np.ndarray,
                     rule: Any, task_path: Optional[str], split: Optional[str] = None,
                     scene_override: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
        if self._active_run_dir is None:
            self.trace_start()
        out_dir = self._active_run_dir

        scene = scene_override if scene_override else extract_scene(model, data)
        try:
            plate_angle = float(data.joint("plate_rotation").qpos[0])
        except Exception:
            plate_angle = float("nan")

        rows, cols = input_grid.shape
        grid_shape = (rows, cols)

        # ---- blocks.csv -> HEAVY DIR (gz if enabled) ----
        if self.write_blocks:
            blocks = []
            for name, obj in scene.items():
                if "position" not in obj:
                    continue
                pos = np.asarray(obj["position"], dtype=float).reshape(-1)
                r, c, dist = _nearest_cell(model, data, pos, grid_shape)
                blocks.append(BlockRecord(name, pos[0], pos[1], pos[2], r, c, dist, obj.get("color", None)))

            bpath = self._heavy_path("blocks.csv")
            if self.gzip_heavy:
                f = gzip.open(bpath + ".gz", "wt", newline="", encoding="utf-8")
            else:
                f = open(bpath, "w", newline="", encoding="utf-8")
            with f:
                w = csv.writer(f)
                w.writerow(["name", "x", "y", "z", "row", "col", "cell_dist", "color"])
                for b in blocks:
                    w.writerow([b.name, b.x, b.y, b.z, b.row, b.col, b.cell_dist, b.color])

        # ---- light files stay in run dir ----
        meta = SnapshotMeta(task_path, split, rule, str(rule), plate_angle, os.path.basename(out_dir), grid_shape)
        with open(os.path.join(out_dir, "Logistics.json"), "w", encoding="utf-8") as f:
            json.dump(_to_list(asdict(meta)), f, indent=2)

        final_grid = scene_to_grid(model, data, rows, cols)
        with open(os.path.join(out_dir, "ARCtask.json"), "w", encoding="utf-8") as f:
            json.dump({
                "input_grid": _to_list(np.asarray(input_grid)),
                "expected_output": _to_list(np.asarray(expected_output)),
                "final_grid": _to_list(np.asarray(final_grid))
            }, f, indent=2)

        self._flush_trace_files(grid_shape=grid_shape)
        self._active_run_dir = None
        self._heavy_run_dir = None
        return out_dir

    # --------------- flushers & utilities ----------------

    def _open_heavy_csv_writer(self, path_no_ext: str, fieldnames: List[str]) -> Tuple[Any, csv.DictWriter]:
        if self.gzip_heavy:
            f = gzip.open(path_no_ext + ".gz", "wt", newline="", encoding="utf-8")
            return f, csv.DictWriter(f, fieldnames=fieldnames)
        else:
            f = open(path_no_ext, "w", newline="", encoding="utf-8")
            return f, csv.DictWriter(f, fieldnames=fieldnames)

    def _flush_trace_files(self, *, grid_shape: Optional[Tuple[int, int]]) -> None:
        if self._active_run_dir is None:
            return

        # trajectory.csv -> HEAVY DIR (gz if enabled)
        if self._traj and self.write_trajectory and self._heavy_run_dir is not None:
            tpath = self._heavy_path("trajectory.csv")
            base = ["t", "x", "y", "z", "qx", "qy", "qz", "qw", "plate_angle", "grip_aperture", "phase"]
            extra_keys = sorted({k for row in self._traj for k in row.keys() if k not in set(base)})
            keys = base + extra_keys
            f, writer = self._open_heavy_csv_writer(tpath, keys)
            try:
                writer.writeheader()
                for row in self._traj:
                    writer.writerow({k: row.get(k, "") for k in keys})
            finally:
                f.close()

        # events.csv (small) -> keep beside meta/grids
        if self._events:
            epath = os.path.join(self._active_run_dir, "events.csv")
            base = ["t", "kind", "label", "x", "y", "z", "qx", "qy", "qz", "qw", "plate_angle", "note"]
            keys = base + ["row", "col", "cell_dist"]
            with open(epath, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for row in self._events:
                    w.writerow({k: row.get(k, "") for k in keys})

    def _reset_buffers(self) -> None:
        self._traj: List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []

    def list_snapshots(self) -> List[str]:
        return sorted([
            os.path.join(self.root_dir, d)
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
            and d != self.heavy_subdir
        ])

    def load_latest(self) -> Optional[Dict[str, Any]]:
        snaps = self.list_snapshots()
        return None if not snaps else self._load_snapshot(snaps[-1])

    def _load_snapshot(self, folder: str) -> Dict[str, Any]:
        out = {"folder": folder}
        for fname in ["Logistics.json", "ARCtask.json", "blocks.csv", "trajectory.csv", "events.csv", "abort.json"]:
            p = os.path.join(folder, fname)
            if os.path.exists(p):
                out[fname] = p
        return out
