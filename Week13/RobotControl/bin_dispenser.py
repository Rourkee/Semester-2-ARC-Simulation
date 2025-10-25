# RobotControl/bin_dispenser.py
import time
import re
import math
import numpy as np
import mujoco

from grid_utils import (
    colour_to_index,          # palette mapper
    grid_to_positions,        # for flat placement targets
    grid_to_layer_position,   # precise per-layer target (k=1 bottom, 2 next, ...)
)

# --- name patterns (module-wide so every method uses the same ones) ---
DISP_PREFIX = re.compile(r"^DISP\d+$")                 # DISP01, DISP2, DISP003...
GRID_PREFIX = re.compile(r"^G\d+\d+(?:_s\d+)?$")       # G11 or stacked: G11_s1, G11_s2, ...

# --- simple 1..9 -> RGB mapping to recolor dispenser blocks ---
_PALETTE = {
    1: (1.0, 0.0, 0.0),  # red
    2: (0.0, 1.0, 0.0),  # green
    3: (0.0, 0.0, 1.0),  # blue
    4: (1.0, 1.0, 0.0),  # yellow
    5: (1.0, 0.0, 1.0),  # magenta
    6: (0.0, 1.0, 1.0),  # cyan
    7: (1.0, 0.5, 0.0),  # orange
    8: (0.5, 0.0, 1.0),  # purple
    9: (1.0, 1.0, 1.0),  # white
}
def _idx_to_rgb(v: int):
    return _PALETTE.get(int(v), (1.0, 1.0, 1.0))

def _body_pos(model, data, body_name):
    return data.body(body_name).xpos.copy()

def _geom_rgba(model, geom_name):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    return model.geom_rgba[gid]

def _set_geom_rgba(model, geom_name, rgba):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    model.geom_rgba[gid][:] = np.array(rgba, dtype=float)

def _free_body_qadr(model, body_name):
    bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jid0 = model.body_jntadr[bid]
    assert model.jnt_type[jid0] == mujoco.mjtJoint.mjJNT_FREE
    return model.jnt_qposadr[jid0]

def _move_free_body(model, data, body_name, xyz, quat=None):
    qadr = _free_body_qadr(model, body_name)
    data.qpos[qadr:qadr+3] = np.asarray(xyz, float).reshape(3)
    if quat is None:
        data.qpos[qadr+3:qadr+7] = np.array([1,0,0,0], float)
    else:
        qq = np.asarray(quat, float).reshape(4)
        data.qpos[qadr+3:qadr+7] = qq / (np.linalg.norm(qq) + 1e-12)
    mujoco.mj_forward(model, data)


class BinDispenserManager:
    """
    - Deletes any block dropped into trash site 'trash_drop'.
    - Detects when a dispenser block is taken from 'dispenser_pick' and
      respawns the next available DISP** block after a short delay (if a pool exists).
    - Provides inventory-delta reconciliation for flat and stacked targets.
    """
    def __init__(self, model, data, robot_core, *, spawn_delay=0.4):
        self.model = model
        self.data  = data
        self.R     = robot_core
        self.spawn_delay = float(spawn_delay)

        # Site handles
        self.trash_site = "trash_drop"
        self.disp_site  = "dispenser_pick"

        # discover all dispenser geoms/bodies (DISP01, DISP02, ...)
        self.pool = self._discover_dispenser_blocks()
        self._active_disp = self._find_block_in_dispenser()  # (geom, body) or None
        self._respawn_at = None

    # ---------- discovery helpers ----------
    def _discover_dispenser_blocks(self):
        names = []
        for i in range(self.model.ngeom):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if not nm:
                continue
            if DISP_PREFIX.match(nm) or (nm.startswith("DISP")):
                bid = self.model.geom_bodyid[i]
                bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                names.append((nm, bname))
        names.sort()
        if not names:
            print("[dispenser] WARNING: no DISP** geoms found.")
        return names

    def _iter_blocks(self):
        """All grid blocks + dispenser blocks as (geom, body)."""
        out = []
        for i in range(self.model.ngeom):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if not nm:
                continue
            if GRID_PREFIX.match(nm) or DISP_PREFIX.match(nm):
                bid = self.model.geom_bodyid[i]
                bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                out.append((nm, bname))
        return out

    # ---------- spatial checks ----------
    def _site_pose(self, site_name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        c = self.data.site_xpos[sid].copy()
        sz = self.model.site_size[sid].copy()
        return c, sz

    def _inside_cylinder_xy(self, p, center, radius):
        return np.linalg.norm(p[:2] - center[:2]) <= float(radius)

    def _find_block_in_dispenser(self, tol=0.06):
        """Return the (geom, body) that currently sits in the dispenser opening, if any."""
        try:
            c, sz = self._site_pose(self.disp_site)
        except Exception:
            return None
        r = float(sz[0]) * 0.8
        best = None; bestd = 1e9
        for (g, b) in self.pool:
            p = _body_pos(self.model, self.data, b)
            d = np.linalg.norm(p[:2] - c[:2])
            if d < r and d < bestd and (c[2] - 0.10) <= p[2] <= (c[2] + 0.10):
                bestd = d; best = (g, b)
        return best

    # ---------- bin logic ----------
    def _delete_block(self, geom_name, body_name):
        """Hide + drop below floor (can't truly remove bodies at runtime)."""
        rgba = _geom_rgba(self.model, geom_name)
        rgba[3] = 0.0
        _set_geom_rgba(self.model, geom_name, rgba)
        _move_free_body(self.model, self.data, body_name, xyz=[0, 0, -1.0])
        if self._active_disp and self._active_disp[0] == geom_name:
            self._active_disp = None

    def _check_trash_bin(self):
        try:
            c, sz = self._site_pose(self.trash_site)
        except Exception:
            return
        rad = float(sz[0]) * 0.9
        for (g, b) in self._iter_blocks():
            p = _body_pos(self.model, self.data, b)
            if self._inside_cylinder_xy(p, c, rad) and p[2] >= c[2] - 0.05:
                self._delete_block(g, b)

    # ---------- dispenser respawn ----------
    def _next_pool_candidate(self, exclude_body):
        """Pick next DISP** that is hidden first (alpha==0), else any far from opening."""
        for (g, b) in self.pool:
            if b == exclude_body:
                continue
            a = _geom_rgba(self.model, g)[3]
            if a < 0.05:
                return (g, b)
        c, sz = self._site_pose(self.disp_site)
        for (g, b) in self.pool:
            if b == exclude_body:
                continue
            p = _body_pos(self.model, self.data, b)
            if np.linalg.norm(p[:2] - c[:2]) > float(sz[0]) * 1.2:
                return (g, b)
        return None

    def _spawn_block_into_dispenser(self):
        # If one is already in the opening, just select it.
        cur = self._find_block_in_dispenser()
        if cur:
            self._active_disp = cur
            return True

        cand = self._next_pool_candidate(exclude_body=(self._active_disp[1] if self._active_disp else None))
        if not cand:
            return False
        g, b = cand
        c, sz = self._site_pose(self.disp_site)
        rgba = _geom_rgba(self.model, g); rgba[3] = 1.0
        _set_geom_rgba(self.model, g, rgba)
        _move_free_body(self.model, self.data, b, xyz=c + np.array([0, 0, -0.04]))
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self._active_disp = (g, b)
        return True

    def _check_dispenser_taken(self):
        now = time.time()
        prev = self._active_disp or self._find_block_in_dispenser()
        self._active_disp = prev
        if not prev:
            if self._respawn_at and now >= self._respawn_at:
                self._spawn_block_into_dispenser()
                self._respawn_at = None
            return

        g, b = prev
        c, sz = self._site_pose(self.disp_site)
        p = _body_pos(self.model, self.data, b)
        # if moved away from opening => schedule respawn
        if np.linalg.norm(p[:2] - c[:2]) > float(sz[0]) * 1.2 or abs(p[2] - c[2]) > 0.12:
            self._respawn_at = time.time() + self.spawn_delay
            self._active_disp = None

    # ---------- public API ----------
    def update(self):
        """Run this every few control steps to handle bin + dispenser."""
        self._check_trash_bin()
        self._check_dispenser_taken()

    def recolor_block(self, geom_name, rgb, alpha=1.0):
        """Set a block geom (grid or DISP**) to new color."""
        r, g_, b_ = [float(x) for x in rgb]
        _set_geom_rgba(self.model, geom_name, (r, g_, b_, float(alpha)))

    def recolor_block_near_hand(self, rgb, alpha=1.0, radius=0.05):
        """Convenience: recolor whichever block is closest to the gripper."""
        hand = self.data.body("panda_hand").xpos.copy()
        best = None; bestd = 1e9
        for (g, b) in self._iter_blocks():
            p = _body_pos(self.model, self.data, b)
            d = np.linalg.norm(p - hand)
            if d < bestd and d <= radius:
                bestd = d; best = g
        if best:
            self.recolor_block(best, rgb, alpha=1.0)
            return best
        return None

    # --- tiny motion helpers for straight-line pick/place using RobotCore ---
    def _safe_hand_quat(self):
        return self.data.body("panda_hand").xquat.copy()

    def _line(self, start, end, steps, quat):
        start = np.asarray(start, float); end = np.asarray(end, float)
        for pos in np.linspace(start, end, int(max(1, steps))):
            self.R.control(pos, quat)
            mujoco.mj_step(self.model, self.data)

    def pick_and_drop(self, pick_pos, drop_pos, *, hover=3, grip_pause=0.25):
        """
        Bin/dispenser sequence that mirrors MovementController kinematics.
        """
        R = self.R

        hand0  = R.data.body("panda_hand").xpos.copy()
        xquat0 = R.data.body("panda_hand").xquat.copy()

        pick_pos = np.array(pick_pos, float).copy()
        drop_pos = np.array(drop_pos, float).copy()

        half = 0.02
        pick_top = pick_pos.copy(); pick_top[2] += half
        drop_top = drop_pos.copy(); drop_top[2] += half

        hover_in  = pick_top.copy();  hover_in[2]  += float(hover)
        hover_out = drop_top.copy();  hover_out[2] += float(hover)

        def _run_segment(path, phase):
            start = path[0]
            if np.linalg.norm(R.data.body("panda_hand").xpos - start) > 1e-3:
                R.correct_to(start, xquat0, threshold=1e-3, steps=60)
            for p in path:
                R.control(p, xquat0)
                mujoco.mj_step(R.model, R.data)
                if R.memory:
                    R.memory.trace_step(R.model, R.data, phase=phase)
                self.update()

        # PICK
        _run_segment(np.linspace(hand0,   hover_in, 1500), "approach_input")
        _run_segment(np.linspace(hover_in, pick_top,  800), "descend_input")
        R.gripper(False); time.sleep(grip_pause if grip_pause is not None else 0.25)
        if R.memory:
            R.memory.log_event(R.model, R.data, kind="pick", label="(bin/pick)", grid_shape=None)
        _run_segment(np.linspace(pick_top, hover_in, 800),  "ascend_input")
        _run_segment(np.linspace(hover_in, hand0,   1500), "return_home")

        # PLACE
        _run_segment(np.linspace(hand0,    hover_out, 1500), "approach_output")
        _run_segment(np.linspace(hover_out, drop_top,  800), "descend_output")
        R.gripper(True); time.sleep(0.2); self.update()
        _run_segment(np.linspace(drop_top, hover_out, 800), "ascend_output")
        _run_segment(np.linspace(hover_out, hand0,    800), "final_home")

        if hasattr(R, "zero_joint_ctrl"):
            R.zero_joint_ctrl()
        R.reset_joints_to_home(steps=400)

    def reconcile_inventory(self,
                            scene: dict,
                            expected_output: np.ndarray,
                            grid_to_positions_fn,
                            grid_shape: tuple[int, int],
                            *,
                            hover: float = 0.12,
                            delete_extras: bool = True) -> None:
        """
        Flat reconcile: recolor in-place, delete extras, add into empty cells.
        """
        import re
        exp = np.asarray(expected_output)
        H, W = map(int, grid_shape)

        GRID_NAME_RE = re.compile(r"^G\d+\d+(?:_s1)?$")

        cur_occ = np.zeros((H, W), dtype=int)
        name_at = [[None for _ in range(W)] for _ in range(H)]
        for nm, obj in scene.items():
            r = int(obj.get("row", -1)); c = int(obj.get("col", -1))
            if 0 <= r < H and 0 <= c < W:
                if name_at[r][c] is None or (GRID_NAME_RE.match(nm) and not GRID_NAME_RE.match(name_at[r][c])):
                    name_at[r][c] = nm
                cur_occ[r, c] = 1

        try:
            bin_c, _ = self._site_pose(self.trash_site)
        except Exception:
            bin_c = np.array([0.0, 0.0, 0.0], float)
        bin_drop = bin_c.copy(); bin_drop[2] += 0.03

        from grid_utils import scene_to_grid
        cur_col = scene_to_grid(self.model, self.data, H, W)

        # recolor in-place
        for r in range(H):
            for c in range(W):
                want = int(exp[r, c])
                if want != 0 and cur_occ[r, c] == 1 and int(cur_col[r, c]) != want:
                    nm = name_at[r][c]
                    if nm is not None:
                        rgb = _idx_to_rgb(want)
                        self.recolor_block(nm, rgb, alpha=1.0)

        mujoco.mj_forward(self.model, self.data)
        self.update()

        # delete extras
        if delete_extras:
            for r in range(H):
                for c in range(W):
                    if exp[r, c] == 0 and cur_occ[r, c] == 1:
                        nm = name_at[r][c]
                        if nm and nm in scene:
                            pick_pos = np.array(scene[nm]["position"], float)
                            self.pick_and_drop(pick_pos, bin_drop, hover=hover)
                            mujoco.mj_forward(self.model, self.data)
                            self.update()

        # refresh occupancy
        from grid_utils import extract_scene as _extract
        scene2 = _extract(self.model, self.data, H, W)
        cur_occ = np.zeros((H, W), dtype=int)
        for nm2, obj2 in scene2.items():
            r = int(obj2.get("row", -1)); c = int(obj2.get("col", -1))
            if 0 <= r < H and 0 <= c < W:
                cur_occ[r, c] = 1

        # add into empty cells
        for r in range(H):
            for c in range(W):
                if exp[r, c] != 0 and cur_occ[r, c] == 0:
                    place_pos = grid_to_positions_fn(self.model, self.data, r, c, H, W)
                    rgb = _idx_to_rgb(exp[r, c])
                    self.pick_recolor_place(place_pos, rgb=rgb, hover=hover, hold_seconds=0.1)
                    mujoco.mj_forward(self.model, self.data)
                    self.update()

    # ---------- stacked inventory reconciliation ----------

    @staticmethod
    def _digits_topdown(v: int) -> list[int]:
        """Encode 0->[], 7->[7], 15->[1,5] (top→down)."""
        v = int(v)
        if v <= 0:
            return []
        return [int(ch) for ch in str(v)]

    def reconcile_inventory_stacked(self,
                                    scene: dict,
                                    expected_output: np.ndarray,
                                    grid_to_positions_fn,
                                    grid_shape: tuple[int, int],
                                    *,
                                    hover: float = 0.12) -> None:
        """
        Make vertical stacks at each cell match expected_output (top→down digits):
          - Recolor to match the desired **suffix** (so existing blocks become the bottom of the final stack).
          - Pop extras from the TOP until lengths match the target suffix.
          - Push the missing **prefix**, placing each new layer on TOP in reverse order
            so the final encoding is correct (e.g., want 15 → add '1' above an existing '5').
        """
        H, W = map(int, grid_shape)
        exp = np.asarray(expected_output)

        # Build per-cell stacks of geoms with their Z (sorted top→down)
        per_cell: list[list[list[tuple[float, int, str]]]] = [[[] for _ in range(W)] for _ in range(H)]
        for nm, obj in scene.items():
            r = int(obj.get("row", -1)); c = int(obj.get("col", -1))
            if not (0 <= r < H and 0 <= c < W):
                continue
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, nm)
            if gid < 0:
                continue
            z = float(obj["position"][2])
            per_cell[r][c].append((z, gid, nm))
        for r in range(H):
            for c in range(W):
                per_cell[r][c].sort(key=lambda t: t[0], reverse=True)  # top→down

        # Utility: current colors (top→down) by reading geom rgba of top item per layer
        def _cur_cols_topdown(layers):
            cols = []
            for group in layers:
                gid0 = group[0][1]
                rgba = self.model.geom_rgba[gid0]
                cols.append(colour_to_index(rgba))
            return cols

        # Bin drop pose
        try:
            bin_c, _ = self._site_pose(self.trash_site)
        except Exception:
            bin_c = np.array([0.0, 0.0, 0.0], float)
        bin_drop = bin_c.copy(); bin_drop[2] += 0.03

        # Group items at nearly same Z into height layers (top→down)
        # Simple clustering per cell using a small tolerance based on observed gaps
        for r in range(H):
            for c in range(W):
                items = per_cell[r][c]
                if not items:
                    continue

                # cluster into layers by Z proximity
                layers: list[list[tuple[float, int, str]]] = []
                z_tol = 1e-3
                cur = []
                z_ref = None
                for z, gid, nm in items:
                    if z_ref is None or abs(z - z_ref) <= z_tol:
                        cur.append((z, gid, nm))
                        if z_ref is None:
                            z_ref = z
                    else:
                        layers.append(cur)
                        cur = [(z, gid, nm)]
                        z_ref = z
                if cur:
                    layers.append(cur)

                # Desired top→down digits & current top→down colors
                want = self._digits_topdown(int(exp[r, c]))
                cur_cols = _cur_cols_topdown(layers)

                m = len(cur_cols)
                n = len(want)

                # Case A: shrink if too tall (pop from top)
                while len(cur_cols) > n:
                    # take the top layer and bin it
                    top_layer = layers.pop(0)
                    top_layer.sort(key=lambda t: t[2])  # stable pick
                    _, _, nm = top_layer[0]
                    pick_pos = np.array(scene[nm]["position"], float)
                    self.pick_and_drop(pick_pos, bin_drop, hover=hover)
                    mujoco.mj_forward(self.model, self.data); self.update()
                    cur_cols.pop(0)

                # Refresh m after possible pops
                m = len(cur_cols)

                # Case B: recolor current layers to match the **desired suffix** (so they become the bottom)
                #   Example: want=[1,5], cur=[5] -> suffix of want len=1 is [5] → no recolor (keep 5 as future bottom)
                suffix = want[-m:] if m > 0 else []
                for k in range(m):
                    if cur_cols[k] != suffix[k]:
                        group = layers[k]
                        group.sort(key=lambda t: t[2])
                        _, _, nm0 = group[0]
                        rgb = _idx_to_rgb(suffix[k])
                        self.recolor_block(nm0, rgb, alpha=1.0)
                        cur_cols[k] = suffix[k]
                mujoco.mj_forward(self.model, self.data); self.update()

                # Case C: push missing **prefix** in reverse, placing each on TOP
                #   Example: want=[1,5], cur(after suffix align)=[5], prefix=[1] → add '1' on top ⇒ [1,5]
                if n > m:
                    prefix = want[:(n - m)]               # top→down elements to add above current
                    for col in reversed(prefix):          # add bottom→top so final order matches
                        # Next layer index is current_stack_len + 1 (1=bottom)
                        layer_idx = len(cur_cols) + 1
                        place_pos = grid_to_layer_position(
                            self.model, self.data, r, c,
                            layer=layer_idx, rows=H, cols=W
                        )
                        rgb = _idx_to_rgb(col)
                        self.pick_recolor_place(place_pos, rgb=rgb, hover=hover, hold_seconds=0.1)
                        mujoco.mj_forward(self.model, self.data); self.update()
                        # new top appears; logically we inserted above → update local mirror
                        cur_cols = [col] + cur_cols

        # done
        return

    # ---- High-level: pick from dispenser -> recolor -> place ----
    def pick_recolor_place(self, target_pos, *, rgb=(1,1,1), hover=0.12, hold_seconds=0.1):
        """
        Controller-agnostic routine using RobotCore straight-line motions.
        """
        R = self.R

        # Ensure a block is ready
        self._spawn_block_into_dispenser()
        found = self._find_block_in_dispenser()
        self._active_disp = found
        if not self._active_disp:
            self.update()
            self._spawn_block_into_dispenser()
            found = self._find_block_in_dispenser()
            self._active_disp = found
        if not self._active_disp:
            print("[dispenser] no block available.")
            return False

        g, b = self._active_disp
        block_center = self.data.body(b).xpos.copy()
        try:
            disp_c, _ = self._site_pose(self.disp_site)
            if not np.isfinite(block_center).all() or np.linalg.norm(block_center[:2] - disp_c[:2]) > 0.08:
                block_center = disp_c + np.array([0, 0, -0.04])
        except Exception:
            pass

        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
        half = 0.02
        if gid >= 0 and int(self.model.geom_type[gid]) == mujoco.mjtGeom.mjGEOM_BOX and len(self.model.geom_size[gid]) >= 3:
            half = float(self.model.geom_size[gid][2])

        hover_in  = block_center.copy(); hover_in[2]  += float(hover)
        touch_in  = block_center.copy(); touch_in[2]  += (half + 0.004)
        down_in   = touch_in.copy();     down_in[2]   -= 0.006

        target_pos = np.array(target_pos, float).copy()
        hover_out  = target_pos.copy();  hover_out[2] += float(hover)
        place_dn   = target_pos.copy();  place_dn[2]  += (half + 0.002)

        xquat = self.data.body("panda_hand").xquat.copy()
        hand0 = self.data.body("panda_hand").xpos.copy()

        # PICK
        for p in np.linspace(hand0, hover_in, 1500):
            R.control(p, xquat); mujoco.mj_step(self.model, self.data)
        for p in np.linspace(hover_in, touch_in, 800):
            R.control(p, xquat); mujoco.mj_step(self.model, self.data)
        for p in np.linspace(touch_in, down_in, 200):
            R.control(p, xquat); mujoco.mj_step(self.model, self.data)

        R.gripper(False); time.sleep(0.2)

        if self._active_disp is not None:
            self._active_disp = None
            self._respawn_at = time.time() + self.spawn_delay
        self.update()

        for p in np.linspace(down_in, hover_in, 800):
            R.control(p, xquat); mujoco.mj_step(self.model, self.data); self.update()
        for p in np.linspace(hover_in, hand0, 1500):
            R.control(p, xquat); mujoco.mj_step(self.model, self.data); self.update()

        # Recolor the exact picked geom
        self.recolor_block(g, rgb, alpha=1.0)
        mujoco.mj_forward(self.model, self.data)
        time.sleep(hold_seconds)

        cur = self.data.body("panda_hand").xpos.copy()
        for p in np.linspace(cur, hover_out, 1500):
            R.control(p, xquat); mujoco.mj_step(self.model, self.data)
        for p in np.linspace(hover_out, place_dn, 800):
            R.control(p, xquat); mujoco.mj_step(self.model, self.data)

        R.gripper(True); time.sleep(0.2); self.update()

        for p in np.linspace(place_dn, hover_out, 800):
            R.control(p, xquat); mujoco.mj_step(self.model, self.data)

        if hasattr(R, "zero_joint_ctrl"):
            R.zero_joint_ctrl()
        R.reset_joints_to_home(steps=400)
        return True
