# RobotControl/bin_dispenser.py
import time
import re
import math
import numpy as np
import mujoco

DISP_PREFIX = re.compile(r"^DISP\d{2}$")   # matches geoms like "DISP01"
GRID_PREFIX = re.compile(r"^G\d+\d+$")     # matches geoms like "G11"

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
    jid0 = model.body_jntadr[bid]              # first joint id for that body
    assert model.jnt_type[jid0] == mujoco.mjtJoint.mjJNT_FREE
    return model.jnt_qposadr[jid0]             # qpos address (7 dof)

def _move_free_body(model, data, body_name, xyz, quat=None):
    qadr = _free_body_qadr(model, body_name)
    data.qpos[qadr:qadr+3] = np.asarray(xyz, float).reshape(3)
    if quat is None:
        data.qpos[qadr+3:qadr+7] = np.array([1,0,0,0], float)  # identity
    else:
        qq = np.asarray(quat, float).reshape(4)
        data.qpos[qadr+3:qadr+7] = qq / (np.linalg.norm(qq) + 1e-12)
    mujoco.mj_forward(model, data)


class BinDispenserManager:
    """
    - Deletes any block dropped into trash site 'trash_drop'.
    - Detects when a dispenser block is taken from 'dispenser_pick' and
      respawns the next available DISP** block after a short delay (if a pool exists).
    - Provides inventory-delta reconciliation:
        * send surplus grid blocks to bin (auto delete)
        * pull blocks from dispenser, recolor in hand, and place to missing cells
    - Also provides a simple pick -> home -> recolor -> place -> home routine.
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
    DISP_PREFIX = re.compile(r"^DISP\d+$")  # allow DISP01, DISP1, DISP002, etc.

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
        c = self.data.site_xpos[sid].copy()          # world-space site position
        sz = self.model.site_size[sid].copy()        # site size 
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
            self.recolor_block(best, rgb, alpha=alpha)
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

    def pick_and_drop(self, pick_pos, drop_pos, *, hover=0.12, grip_pause=0.25):
        """
        Approach pick_pos, grasp, lift, move to drop_pos, release, retreat, home.
        """
        xquat = self._safe_hand_quat()
        hand0 = self.data.body("panda_hand").xpos.copy()

        above_pick = np.array(pick_pos, float).copy(); above_pick[2] += hover
        above_drop = np.array(drop_pos, float).copy(); above_drop[2] += hover
        pick_pos   = np.array(pick_pos, float)
        drop_pos   = np.array(drop_pos, float)

        # approach & grip
        self._line(hand0,      above_pick, 1500, xquat)
        self._line(above_pick, pick_pos,   300, xquat)
        self.R.gripper(False); time.sleep(0.25 if grip_pause is None else grip_pause)

        # lift, move over, descend, release
        self._line(pick_pos,   above_pick, 300, xquat)
        self._line(above_pick, above_drop, 800, xquat)
        self._line(above_drop, drop_pos,   300, xquat)
        self.R.gripper(True);  time.sleep(0.25 if grip_pause is None else grip_pause)

        # retreat and home
        self._line(drop_pos,   above_drop, 200, xquat)
        if hasattr(self.R, "zero_joint_ctrl"):
            self.R.zero_joint_ctrl()
        self.R.reset_joints_to_home(steps=400)

    def reconcile_inventory(self,
                            scene: dict,
                            expected_output: np.ndarray,
                            grid_to_positions_fn,
                            grid_shape: tuple[int, int],
                            *,
                            hover: float = 0.12) -> None:
        """
        Make the physical scene match expected_output by:
          - sending surplus grid blocks to trash (bin will delete),
          - pulling blocks from dispenser, recoloring while held, and placing in missing cells.

        Parameters
        ----------
        scene : dict from extract_scene(...)
        expected_output : HxW numpy array of color indices
        grid_to_positions_fn : callable(model, data, r, c, H, W) -> (x,y,z)
        grid_shape : (H, W)
        hover : hover height for straight-line moves
        """
        exp = np.asarray(expected_output)
        H, W = map(int, grid_shape)

        # build current occupancy + name lookup per cell
        cur = np.zeros((H, W), dtype=int)
        name_at = [[None for _ in range(W)] for _ in range(H)]
        for nm, obj in scene.items():
            r = int(obj.get("row", -1)); c = int(obj.get("col", -1))
            if 0 <= r < H and 0 <= c < W:
                cur[r, c] = 1
                name_at[r][c] = nm

        # bin drop pose
        try:
            bin_c, _ = self._site_pose(self.trash_site)
        except Exception:
            bin_c = np.array([0.0, 0.0, 0.0], float)
        bin_drop = bin_c.copy(); bin_drop[2] += 0.03

        # 1) remove extras (expected==0 and cur==1)
        for r in range(H):
            for c in range(W):
                if exp[r, c] == 0 and cur[r, c] == 1:
                    nm = name_at[r][c]
                    if nm and nm in scene:
                        pick_pos = np.array(scene[nm]["position"], float)
                        self.pick_and_drop(pick_pos, bin_drop, hover=hover)
                        mujoco.mj_forward(self.model, self.data)
                        self.update()  # triggers delete if in bin

        # 2) add missing (expected!=0 and cur==0)
        for r in range(H):
            for c in range(W):
                if exp[r, c] != 0 and cur[r, c] == 0:
                    place_pos = grid_to_positions_fn(self.model, self.data, r, c, H, W)
                    rgb = _idx_to_rgb(exp[r, c])
                    self.pick_recolor_place(place_pos, rgb=rgb, hover=hover, hold_seconds=0.1)
                    mujoco.mj_forward(self.model, self.data)
                    self.update()  # allow respawn scheduling

    # ---- High-level: pick from dispenser -> home -> recolor -> place -> home ----
    def pick_recolor_place(self, target_pos, *, rgb=(1,1,1), hover=0.12, hold_seconds=0.2):
        """
        Minimal, controller-agnostic routine using RobotCore straight-line motions.
        target_pos: (x,y,z) world position to place the block center.
        rgb: desired final color while held.
        """
        R = self.R

        # Ensure a block is available in the dispenser
        if not self._active_disp:
            self._spawn_block_into_dispenser()
        self._active_disp = self._find_block_in_dispenser()
        if not self._active_disp:
            print("[dispenser] no block available.")
            return False

        # Use the ACTUAL block pose 
        g, b = self._active_disp
        block_center = self.data.body(b).xpos.copy()

        # if something is off, use site-based guess
        try:
            disp_c, _ = self._site_pose(self.disp_site)
            if not np.isfinite(block_center).all() or np.linalg.norm(block_center[:2] - disp_c[:2]) > 0.08:
                block_center = disp_c + np.array([0, 0, -0.04])
        except Exception:
            pass

        xquat = self.data.body("panda_hand").xquat.copy()

        # Approach above dispenser
        above = block_center.copy(); above[2] += hover
        hand0 = R.data.body("panda_hand").xpos.copy()
        for pos in np.linspace(hand0, above, 400):
            R.control(pos, xquat); mujoco.mj_step(self.model, self.data)

        # Descend and grip
        for pos in np.linspace(above, block_center, 200):
            R.control(pos, xquat); mujoco.mj_step(self.model, self.data)
        R.gripper(False)
        time.sleep(0.25)

        # Lift
        lift = block_center.copy(); lift[2] += hover
        for pos in np.linspace(block_center, lift, 200):
            R.control(pos, xquat); mujoco.mj_step(self.model, self.data)

        # Home with block
        if hasattr(R, "zero_joint_ctrl"):
            R.zero_joint_ctrl()
        R.reset_joints_to_home(steps=400)

        # Recolor while held
        self.recolor_block_near_hand(rgb, alpha=1.0, radius=0.07)
        time.sleep(hold_seconds)

        # Move to place target (simple straight-line)
        cur = R.data.body("panda_hand").xpos.copy()
        tgt_up = np.array(target_pos, float).copy(); tgt_up[2] += hover
        tgt_dn = np.array(target_pos, float).copy()
        for pos in np.linspace(cur, tgt_up, 400):
            R.control(pos, xquat); mujoco.mj_step(self.model, self.data)
        for pos in np.linspace(tgt_up, tgt_dn, 200):
            R.control(pos, xquat); mujoco.mj_step(self.model, self.data)

        # Release and retreat, then home
        R.gripper(True)
        time.sleep(0.25)
        for pos in np.linspace(tgt_dn, tgt_up, 200):
            R.control(pos, xquat); mujoco.mj_step(self.model, self.data)
        if hasattr(R, "zero_joint_ctrl"):
            R.zero_joint_ctrl()
        R.reset_joints_to_home(steps=400)
        return True
