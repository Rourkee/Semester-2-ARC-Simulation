# rotation_controller.py
import math
import time
import numpy as np
import mujoco


class RotationController:
    """
    Minimal plate rotation via a single knob, using RobotCore.
    Modes:
      - Angle-based: angle_rad -> rotate by that angle.
      - Goal-based: until_done_fn(model, data) -> rotate until predicate true.
    Post-actions:
      - release_in_place
      - release_and_hover
      - release_and_home
      - release_hover_then_home
    """

    def __init__(self, robot):
        self.R = robot  # RobotCore

    # ----------------------- helpers -----------------------

    def _get_knob(self):
        """Pick a single knob target from RobotCore."""
        R = self.R
        targets = R.find_knob_targets()
        if "primary" not in targets:
            print("[spin] no knob target found")
            return None
        t = targets["primary"]
        pos = R.site_pos(t["name"]) if t["type"] == "site" else R.geom_pos(t["name"])
        if pos is None:
            print(f"[spin] knob '{t['name']}' not found")
            return None
        return {"name": t["name"], "pos": pos}

    @staticmethod
    def _line(a, b, n):
        a = np.asarray(a, float); b = np.asarray(b, float)
        return np.linspace(a, b, int(max(1, n)))

    def _move(self, path, quat, phase):
        R = self.R
        for p in path:
            R.control(p, quat)
            mujoco.mj_step(R.model, R.data)
            if R.memory:
                R.memory.trace_step(R.model, R.data, phase=phase)

    def _post_action(self, mode, last_pos, last_quat, hover_z, hold_seconds):
        """Release, then do the requested exit behavior."""
        R = self.R
        # always release first
        R.gripper(True)
        time.sleep(0.25)

        if mode == "release_in_place":
            return {"end_pos": last_pos, "end_quat": last_quat}

        # ascend straight up to hover
        start = last_pos.copy()
        end = last_pos.copy(); end[2] = hover_z
        self._move(self._line(start, end, 200), last_quat, "ascend_vertical")

        if mode == "release_and_hover":
            return {"end_pos": end, "end_quat": last_quat}

        if mode == "release_hover_then_home":
            if hold_seconds and hold_seconds > 0:
                time.sleep(hold_seconds)
            # fall-through to home

        if mode in ("release_hover_then_home", "release_and_home"):
            if hasattr(R, "zero_joint_ctrl"):
                R.zero_joint_ctrl()
            R.reset_joints_to_home(steps=400)
            return {
                "end_pos": R.data.body("panda_hand").xpos.copy(),
                "end_quat": R.data.body("panda_hand").xquat.copy(),
            }

        raise ValueError(f"Unknown seg_post_action '{mode}'")

    # ----------------------- main API -----------------------

    def spin_plate_knob(
        self,
        angle_rad: float | None = None,
        *,
        until_done_fn=None,       # callable(model, data) -> bool
        done_check_every=5,       # (ignored; kept for API compatibility)
        grip_height_offset=0.0,
        arc_inner_iters=2,
        dwell=0.0,
        post_action="release_and_home",
        hold_seconds=0.4,
        stall_tol=np.deg2rad(0.05),
        stall_iters=2000,
        direction_sign: float = 1.0,   # preferred direction in goal-mode (+1 ccw, -1 cw)
    ):
        """
        If angle_rad is None and until_done_fn is provided -> goal-mode rotation.
        Otherwise rotate by angle_rad (angle-mode).
        """
        R = self.R
        k = self._get_knob()
        if not k:
            return
        knob_name, knob_pos = k["name"], k["pos"]

        # geometry/orientation
        plate_center, _, z_top = R.plate_geom_info()
        grip_height = float(knob_pos[2] + grip_height_offset)
        hover_z = grip_height + 0.15

        hand0 = R.data.body("panda_hand").xpos.copy()
        xquat0 = R.data.body("panda_hand").xquat.copy()

        radius = float(np.linalg.norm(knob_pos[:2] - plate_center[:2]))
        vertical_gap = float(abs(z_top - grip_height))
        pitch = math.atan2(vertical_gap, max(radius, 1e-6)) * 0.5
        xquat_app = R._apply_ypr(xquat0, yaw=0.0, pitch=pitch, roll=0.0)

        # goal-mode early exit: if already solved, just do the post-action without touching the knob
        if angle_rad is None and callable(until_done_fn) and until_done_fn(R.model, R.data):
            return self._post_action(post_action, hand0, xquat0, hover_z, hold_seconds)

        # approach & grasp
        hover_knob = knob_pos.copy(); hover_knob[2] = hover_z
        grip_knob  = knob_pos.copy(); grip_knob[2]  = grip_height
        self._move(self._line(hand0, hover_knob, 600), xquat_app, "approach")
        self._move(self._line(hover_knob, grip_knob, 400), xquat_app, "descend")
        R.gripper(False); time.sleep(0.25)

        # rotation loop
        plate_start = R.plate_angle()
        plate_goal = None if angle_rad is None else (plate_start + float(angle_rad))

        # tolerance and step sizing from knob radius (fallbacks are modest)
        gid = mujoco.mj_name2id(R.model, mujoco.mjtObj.mjOBJ_GEOM, knob_name)
        stem_rad = float(R.model.geom_size[gid][0]) if gid >= 0 else 0.010
        angle_tol = float(np.clip(0.5 * stem_rad / max(radius, 1e-6), np.deg2rad(0.25), np.deg2rad(2.0)))
        step_cap  = float(np.clip(4.0 * angle_tol, np.deg2rad(0.5), np.deg2rad(5.0)))

        rel0_xy = (knob_pos[:2] - plate_center[:2]).copy()
        last_progress = plate_start
        stalled = 0

        while True:
            plate_now = R.plate_angle()

            # stops
            if plate_goal is None:
                if callable(until_done_fn) and until_done_fn(R.model, R.data):
                    break
                dtheta = float(direction_sign) * step_cap
            else:
                err = R._wrap_to_pi(plate_goal - plate_now)
                if abs(err) <= angle_tol:
                    break
                dtheta = math.copysign(min(abs(err), step_cap), err)

            # target along the circle; keep a simple yaw that follows progress
            theta_cmd = plate_now + dtheta
            theta_rel = theta_cmd - plate_start
            phi = math.atan2(rel0_xy[1], rel0_xy[0]) + theta_rel
            pos_des = np.array(
                [plate_center[0] + radius * math.cos(phi),
                 plate_center[1] + radius * math.sin(phi),
                 grip_height], dtype=float)
            yaw_rel = R._wrap_to_pi(theta_rel)
            xquat_d = R._apply_ypr(xquat0, yaw=yaw_rel, pitch=pitch, roll=0.0)

            for _ in range(int(max(1, arc_inner_iters))):
                R.control(pos_des, xquat_d)
                mujoco.mj_step(R.model, R.data)
                if R.memory:
                    R.memory.trace_step(R.model, R.data, phase="rotate_arc")
                if dwell:
                    time.sleep(dwell)

            # stall guard
            if abs(plate_now - last_progress) < max(float(stall_tol), step_cap * 0.1):
                stalled += 1
                if stalled > int(max(1000, stall_iters)):
                    print("[spin] stalled; stopping early")
                    break
            else:
                stalled = 0
                last_progress = plate_now

        # exit behavior (release -> hover/home as requested)
        return self._post_action(post_action,
                                 last_pos=pos_des,
                                 last_quat=xquat_d,
                                 hover_z=hover_z,
                                 hold_seconds=hold_seconds)
