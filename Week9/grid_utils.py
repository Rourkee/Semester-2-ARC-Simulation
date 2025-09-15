from __future__ import annotations
import re
from typing import Dict, Optional, Tuple, Iterable
import numpy as np
import mujoco

# =========================
# Configuration & Globals
# =========================

# Grid row indexing convention:
#  - 'top'    : ARC-style (row index increases downward in screen/plate Y)
#  - 'bottom' : Cartesian-style (row index increases upward)
_GRID_ROW0_AT = 'top'

# Orientation policy for blocks when (re)placing them:
#  - 'identity' : set quaternion to [1,0,0,0]
#  - 'plate'    : align to the plate's current world orientation
_BLOCK_ORIENTATION_MODE = 'identity'

# Cache of per-cell Z (so subsequent placements reuse the measured height)
_CELL_Z_CACHE: Optional[np.ndarray] = None


def set_grid_convention(row0_at: str = 'top') -> None:
    """
    Choose how row indices are interpreted.
    'top' (default): row 0 is visually at the top (downward = +row)  [ARC default]
    'bottom'       : row 0 is at the bottom (upward = +row)
    """
    global _GRID_ROW0_AT
    row0_at = (row0_at or 'top').strip().lower()
    if row0_at not in ('top', 'bottom'):
        raise ValueError("row0_at must be 'top' or 'bottom'")
    _GRID_ROW0_AT = row0_at


def set_block_orientation_mode(mode: str = 'identity') -> None:
    """
    Set the orientation policy used when placing blocks.
    'identity' : quaternion = [1,0,0,0]
    'plate'    : quaternion copied from the current plate/world anchor
    """
    global _BLOCK_ORIENTATION_MODE
    mode = (mode or 'identity').strip().lower()
    if mode not in ('identity', 'plate'):
        raise ValueError("mode must be 'identity' or 'plate'")
    _BLOCK_ORIENTATION_MODE = mode


# =========================
# Colors
# =========================
# 0 is now a VISIBLE GREY block (alpha=1.0). No special "hidden" treatment for 0.
# --- Colors: 0 is "no color" / empty ---
COLOUR_MAP = {
    0: None,              # EMPTY (do not try to match this in colour_to_index)
    1: [1, 0, 0, 1],
    2: [0, 1, 0, 1],
    3: [0, 0, 1, 1],
    4: [1, 1, 0, 1],
    5: [1, 0, 1, 1],
    6: [0, 1, 1, 1],
    7: [1, 0.5, 0, 1],
    8: [0.5, 0, 1, 1],
    9: [1, 1, 1, 1],
}

def colour_to_index(rgba, tol: float = 0.05) -> int:
    """
    Map a MuJoCo RGBA back to nearest palette index.
    - Skip palette entries that are None (e.g., 0).
    - If alpha ~ 0, treat as empty (0).
    """
    if rgba is None or float(rgba[3]) <= 0.01:
        return 0
    rgb = np.asarray(rgba[:3], dtype=float)
    best, bestd = 0, float("inf")
    for k, col in COLOUR_MAP.items():
        if col is None:            # don't compare against "empty"
            continue
        d = float(np.linalg.norm(rgb - np.asarray(col[:3], dtype=float)))
        if d < bestd:
            best, bestd = k, d
    return int(best)


# =========================
# MuJoCo name helpers
# =========================

def _body_exists(model, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0

def _geom_exists(model, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0

def _site_exists(model, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0

def _anchor_body_name(model) -> str:
    """Prefer plate child if present, else plate root, else world."""
    for nm in ("spinning_plate", "tray", "plate_root"):
        if _body_exists(model, nm):
            return nm
    # id 0 is "world"
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 0)  # type: ignore


# =========================
# Pose / frame utilities
# =========================

def _plate_pose(model, data) -> Tuple[np.ndarray, np.ndarray]:
    """Return (world_position, world_rotation_matrix 3x3) of the anchor body."""
    bname = _anchor_body_name(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
    pos = data.xpos[bid].copy()
    R = data.xmat[bid].reshape(3, 3).copy()
    return pos, R

def _plate_world_quat(model, data) -> np.ndarray:
    """World quaternion (w,x,y,z) of the anchor body."""
    bname = _anchor_body_name(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
    return data.xquat[bid].copy()

def _plate_top_z(model, data) -> float:
    """
    Approx plate top Z from named geoms ('plate_geom' or 'tray_geom').
    Falls back to 0.07 if unknown.
    """
    for g in ("plate_geom", "tray_geom"):
        if _geom_exists(model, g):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            gtype = int(model.geom_type[gid])
            if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                half_h = float(model.geom_size[gid][1])
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                half_h = float(model.geom_size[gid][2])
            else:
                half_h = float(model.geom_size[gid][1])
            bid = int(model.geom_bodyid[gid])
            return float(data.xpos[bid][2] + half_h)
    return 0.07


def _infer_shape_from_names(model) -> Tuple[int, int]:
    """Parse geom names 'Grc' to infer (rows, cols). Returns at least (1,1)."""
    rows = cols = 0
    rgx = re.compile(r"^G(\d+)(\d+)$")
    for gid in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not nm:
            continue
        m = rgx.match(nm)
        if m:
            r = int(m.group(1)); c = int(m.group(2))
            rows = max(rows, r); cols = max(cols, c)
    return max(rows, 1), max(cols, 1)


# =========================
# Grid parameterization
# =========================

def _pitch_from_lines(model, data, prefix: str, axis_hat: np.ndarray) -> Optional[float]:
    """Return spacing between line 0 and 1 along the given axis (|dot(diff,axis)|)."""
    try:
        id0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}0")
        id1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}1")
        if id0 >= 0 and id1 >= 0:
            diff = data.geom_xpos[id1] - data.geom_xpos[id0]
            return float(abs(np.dot(diff, axis_hat)))
    except Exception:
        pass
    return None

def _grid_params(model, data, rows: int, cols: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Return (origin, v_col, v_row, z_center) using plate pose and grid line spacing.
    Robust even when some blocks (zeros) are not spawned.
    """
    pc, R = _plate_pose(model, data)
    ex, ey = R[:, 0], R[:, 1]  # plate +X (cols), +Y (rows)

    # Prefer spacing from the grid line geoms; these always exist.
    x_pitch = _pitch_from_lines(model, data, "grid_v_", ex)
    y_pitch = _pitch_from_lines(model, data, "grid_h_", ey)

    # Last-ditch fallback if lines missing (e.g., custom XML): keep previous heuristics
    if x_pitch is None:
        try:
            b12 = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "G12_b")]
            b11 = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "G11_b")]
            x_pitch = float(abs(np.dot(b12 - b11, ex)))
        except Exception:
            x_pitch = 0.10  # conservative fallback
    if y_pitch is None:
        try:
            b21 = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "G21_b")]
            b11 = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "G11_b")]
            y_pitch = float(abs(np.dot(b21 - b11, ey)))
        except Exception:
            y_pitch = 0.10

    v_col = ex * x_pitch
    v_row = (ey * y_pitch) if _GRID_ROW0_AT == 'bottom' else (-ey * y_pitch)

    # Center the grid on the plate center
    origin = pc - 0.5 * (cols - 1) * v_col - 0.5 * (rows - 1) * v_row

    # Z default: measured median block center, else plate top + half block
    z_default = _measured_block_center_z(model, data, default=_plate_top_z(model, data) + 0.03)
    origin = origin.copy()
    origin[2] = z_default
    return origin, v_col, v_row, z_default

def _measured_block_center_z(model, data, default: float = 0.10) -> float:
    zs = []
    for gid in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if nm and nm.startswith("G"):
            zs.append(float(data.geom_xpos[gid][2]))
    if zs:
        return float(np.median(zs))
    return float(default)


def cell_pitch_xy(model, data, rows: int, cols: int) -> float:
    """
    Return the scalar cell pitch in the plate plane (meters).
    Uses the average of |v_col| and |v_row|.
    """
    origin, v_col, v_row, _ = _grid_params(model, data, rows, cols)
    px = float(np.linalg.norm(v_col[:2]))
    py = float(np.linalg.norm(v_row[:2]))
    if px <= 1e-9 and py <= 1e-9:
        return 0.10
    if px <= 1e-9:
        return py
    if py <= 1e-9:
        return px
    return 0.5 * (px + py)


# =========================
# Z cache
# =========================

def cache_cell_heights_from_scene(model, data, rows: int, cols: int) -> None:
    """Populate per-cell Z cache from current block positions (median fill)."""
    global _CELL_Z_CACHE
    z_grid = np.full((rows, cols), np.nan, dtype=float)

    origin, v_col, v_row, _ = _grid_params(model, data, rows, cols)
    vcol_hat = v_col[:2] / (np.linalg.norm(v_col[:2]) + 1e-12)
    vrow_hat = v_row[:2] / (np.linalg.norm(v_row[:2]) + 1e-12)
    pitch_x = np.linalg.norm(v_col[:2]); pitch_y = np.linalg.norm(v_row[:2])

    for gid in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not nm or not nm.startswith("G"):
            continue
        pos = data.geom_xpos[gid]
        dxy = pos[:2] - origin[:2]
        c = int(round(np.dot(dxy, vcol_hat) / (pitch_x + 1e-12)))
        r = int(round(np.dot(dxy, vrow_hat) / (pitch_y + 1e-12)))
        if 0 <= r < rows and 0 <= c < cols:
            z_grid[r, c] = float(pos[2])

    if np.isnan(z_grid).any():
        existing = z_grid[~np.isnan(z_grid)]
        fill = float(np.median(existing)) if existing.size else (_plate_top_z(model, data) + 0.03)
        z_grid = np.where(np.isnan(z_grid), fill, z_grid)

    _CELL_Z_CACHE = z_grid


# =========================
# Free joint setter
# =========================

def _set_free_body_pose(model, data, body_name: str, pos: np.ndarray, quat: Optional[np.ndarray] = None) -> bool:
    """Set a free joint body pose; return True if successful."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return False
    jadr = int(model.body_jntadr[bid])
    if jadr < 0 or int(model.jnt_type[jadr]) != mujoco.mjtJoint.mjJNT_FREE:
        return False
    qadr = int(model.jnt_qposadr[jadr])
    data.qpos[qadr:qadr+3] = np.asarray(pos, dtype=float).reshape(3)
    if quat is None:
        data.qpos[qadr+3:qadr+7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        q = np.asarray(quat, dtype=float).reshape(4)
        data.qpos[qadr+3:qadr+7] = q
    return True


# =========================
# Public API
# =========================

def grid_to_positions(model, data, row: int, col: int,
                      rows: Optional[int] = None, cols: Optional[int] = None) -> np.ndarray:
    """
    World position for a given (row, col) center.
    Uses plate pose for XY; Z from cache (if available) or default center Z.
    """
    if rows is None or cols is None:
        r_i, c_i = _infer_shape_from_names(model)
        rows = rows if rows is not None else r_i
        cols = cols if cols is not None else c_i

    origin, v_col, v_row, zc = _grid_params(model, data, rows, cols)
    p = origin + col * v_col + row * v_row

    global _CELL_Z_CACHE
    if _CELL_Z_CACHE is not None and 0 <= row < _CELL_Z_CACHE.shape[0] and 0 <= col < _CELL_Z_CACHE.shape[1]:
        p[2] = float(_CELL_Z_CACHE[row, col])
    else:
        p[2] = zc
    return p.astype(float)


def apply_input_grid(model, data, input_grid: np.ndarray) -> None:
    rows, cols = input_grid.shape
    cache_cell_heights_from_scene(model, data, rows, cols)

    plate_q = _plate_world_quat(model, data) if _BLOCK_ORIENTATION_MODE == 'plate' else None
    graveyard = np.array([-10.0, -10.0, 0.02], dtype=float)  # far away & low collision risk

    for r in range(rows):
        for c in range(cols):
            name = f"G{r+1}{c+1}"
            body = f"{name}_b"
            gid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            val  = int(input_grid[r, c])

            if val == 0:
                # If this geom exists (e.g., in a full-grid world), hide + park it.
                if gid >= 0:
                    model.geom_rgba[gid] = np.array([0, 0, 0, 0], dtype=np.float32)  # fully transparent
                _set_free_body_pose(model, data, body, graveyard, quat=None)
                continue

            # Nonzero: set color and place at its cell
            if gid >= 0:
                col = COLOUR_MAP.get(val)
                if col is not None:
                    model.geom_rgba[gid] = np.array(col, dtype=np.float32)

            pos = grid_to_positions(model, data, r, c, rows, cols)
            if not _set_free_body_pose(model, data, body, pos, quat=plate_q):
                if gid >= 0:
                    model.geom_pos[gid] = pos

    mujoco.mj_forward(model, data)



def extract_scene(model, data, rows: Optional[int] = None, cols: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Return a dict: geom name -> {'row': r, 'col': c, 'position': (3,)}.
    r/c=-1 if a block is outside the grid footprint.
    """
    if rows is None or cols is None:
        r_i, c_i = _infer_shape_from_names(model)
        rows = rows if rows is not None else r_i
        cols = cols if cols is not None else c_i

    origin, v_col, v_row, _ = _grid_params(model, data, rows, cols)
    vcol_hat = v_col[:2] / (np.linalg.norm(v_col[:2]) + 1e-12)
    vrow_hat = v_row[:2] / (np.linalg.norm(v_row[:2]) + 1e-12)
    pitch_x = np.linalg.norm(v_col[:2]); pitch_y = np.linalg.norm(v_row[:2])

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for gid in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        # --- minimal change: accept G** and DISP** as grid-occupying blocks ---
        if not nm or (not nm.startswith("G") and not nm.startswith("DISP")):
            continue
        pos = data.geom_xpos[gid].copy()
        dxy = pos[:2] - origin[:2]
        c = int(round(np.dot(dxy, vcol_hat) / (pitch_x + 1e-12)))
        r = int(round(np.dot(dxy, vrow_hat) / (pitch_y + 1e-12)))
        if 0 <= r < rows and 0 <= c < cols:
            out[nm] = {"row": int(r), "col": int(c), "position": pos}
        else:
            out[nm] = {"row": -1, "col": -1, "position": pos}
    return out


def scene_to_grid(model, data, rows: Optional[int] = None, cols: Optional[int] = None) -> np.ndarray:
    """
    Build an integer grid from the current scene by snapping blocks to nearest cells.
    Colors are mapped to COLOUR_MAP.
    """
    if rows is None or cols is None:
        r_i, c_i = _infer_shape_from_names(model)
        rows = rows if rows is not None else r_i
        cols = cols if cols is not None else c_i

    grid = np.zeros((rows, cols), dtype=int)
    origin, v_col, v_row, _ = _grid_params(model, data, rows, cols)
    vcol_hat = v_col[:2] / (np.linalg.norm(v_col[:2]) + 1e-12)
    vrow_hat = v_row[:2] / (np.linalg.norm(v_row[:2]) + 1e-12)
    pitch_x = np.linalg.norm(v_col[:2]); pitch_y = np.linalg.norm(v_row[:2])

    for gid in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        # --- minimal change: accept G** and DISP** as grid-occupying blocks ---
        if not nm or (not nm.startswith("G") and not nm.startswith("DISP")):
            continue
        pos = data.geom_xpos[gid]
        dxy = pos[:2] - origin[:2]
        c = int(round(np.dot(dxy, vcol_hat) / (pitch_x + 1e-12)))
        r = int(round(np.dot(dxy, vrow_hat) / (pitch_y + 1e-12)))
        if 0 <= r < rows and 0 <= c < cols:
            rgba = model.geom_rgba[gid]
            grid[r, c] = colour_to_index(rgba)
    return grid


# =========================
# Goal / completion helpers
# =========================

def goal_positions(model, data, goal: Dict[str, Dict[str, int]], grid_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """
    Resolve a name->(row,col) goal mapping into world target positions.
    Returns: { geom_name -> (3,) target world position }
    """
    rows, cols = grid_shape
    out: Dict[str, np.ndarray] = {}
    for name, rc in goal.items():
        r, c = int(rc["row"]), int(rc["col"])
        out[name] = grid_to_positions(model, data, r, c, rows, cols)
    return out


def is_block_at_target(pos_now: np.ndarray, pos_target: np.ndarray, xy_tol_m: float) -> bool:
    """
    True if current position is within tolerance of target in the XY plane.
    Z is handled via cached heights in target generation.
    """
    dxy = np.linalg.norm((pos_now[:2] - pos_target[:2]).astype(float))
    return bool(dxy <= float(xy_tol_m))


def is_goal_reached(model, data,
                    goal: Dict[str, Dict[str, int]],
                    grid_shape: Tuple[int, int],
                    pos_tol_cells: float = 0.10) -> Tuple[bool, Dict[str, float]]:
    """
    Check whether all blocks reached their targets within tolerance.

    Args:
      goal: mapping { "Grc": {"row": r, "col": c}, ... }
      grid_shape: (rows, cols)
      pos_tol_cells: tolerance as a fraction of cell pitch (default 0.10 = 10%)

    Returns:
      (done, errors) where:
        - done: True iff all named blocks are within tolerance
        - errors: per-block XY error (meters) for diagnostics
    """
    rows, cols = grid_shape
    pitch = cell_pitch_xy(model, data, rows, cols)
    xy_tol_m = float(pos_tol_cells) * float(pitch or 0.10)

    tgt = goal_positions(model, data, goal, grid_shape)
    scn = extract_scene(model, data, rows, cols)

    errors: Dict[str, float] = {}
    all_ok = True
    for name, target in tgt.items():
        if name not in scn:
            all_ok = False
            errors[name] = float("inf")
            continue
        now = scn[name]["position"]
        err = float(np.linalg.norm((now[:2] - target[:2]).astype(float)))
        errors[name] = err
        if err > xy_tol_m:
            all_ok = False

    return all_ok, errors


# Optional: rotation utilities (useful to "spin until done")
def plate_rotation_error(model, data, target_angle_rad: float) -> float:
    """
    If a 'plate_rotation' hinge exists, return the shortest signed angle
    from current angle to target_angle_rad, wrapped to [-pi, pi].
    If no hinge, returns 0.0.
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_rotation")
    if jid < 0:
        return 0.0
    qadr = int(model.jnt_qposadr[jid])
    q = float(data.qpos[qadr])  # hinge angle
    diff = float(target_angle_rad) - q
    # wrap to [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff
