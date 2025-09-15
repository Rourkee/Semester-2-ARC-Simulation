##### rotation_controller.py
import math
import time
import numpy as np
import mujoco


class RotationController:
    """
    Rotation controller using RobotCore services.

    Modes:
      1) Angle-based: angle_rad -> rotate by that angle (stop immediately when reached).
      2) Goal-based: until_done_fn(model, data) -> keep rotating until it returns True.

    Notes:
      - Prefers site 'knob_grasp' / geom 'knob_handle' automatically; falls back to 'knob_stem'.
    """

    def __init__(self, robot):
        self.R = robot  # RobotCore

    def spin_plate_knob(
        self,
        angle_rad: float | None = None,
        *,
        until_done_fn=None,       # callable(model, data) -> bool
        done_check_every=30,      # iterations between done checks
        grip_height_offset=0.0,
        arc_inner_iters=8,
        dwell=0.0,
        post_action="release_and_home",
        hold_seconds=0.4,
        stall_tol=1e-4,
        stall_iters=2000,
    ):
        """
        If angle_rad is None and until_done_fn is provided -> rotate until until_done_fn() is True.
        Otherwise rotate by angle_rad (single knob only).

        done_check_every: how often (iterations) to query until_done_fn to stop early when solved.
        """
        R = self.R
        targets = R.find_knob_targets()

        # pick single knob
        if "primary" not in targets:
            print("[spin] no knob target found; aborting")
            return
        prim = targets["primary"]
        knob_name = prim["name"]

        def _segment(knob_name: str, seg_angle: float | None, phase_tag: str, seg_post_action: str):
            """
            Perform one contiguous rotation segment on a single knob.
            If seg_angle is None, drive until until_done_fn() returns True.
            """
            # Fetch knob pose live each time (plate rotates)
            def _knob_pos():
                if prim["type"] == "site" and knob_name == prim["name"]:
                    return R.site_pos(knob_name)
                return R.geom_pos(knob_name)

            hand0 = R.data.body("panda_hand").xpos.copy()
            xquat0 = R.data.body("panda_hand").xquat.copy()

            knob_pos = _knob_pos()
            if knob_pos is None:
                print(f"[spin] knob target '{knob_name}' not found; skipping segment '{phase_tag}'")
                return None
            plate_center, r_plate, z_top = R.plate_geom_info()
            grip_height = float(knob_pos[2] + grip_height_offset)
            hover_knob = knob_pos.copy(); hover_knob[2] = grip_height + 0.15
            grip_knob = knob_pos.copy(); grip_knob[2] = grip_height

            radius = float(np.linalg.norm(knob_pos[:2] - plate_center[:2]))
            vertical_gap = float(abs(z_top - grip_height))
            pitch = math.atan2(vertical_gap, max(radius, 1e-6)) * 0.5  # slight pitch inward to resist slip
            roll = 0.0
            xquat_app = R._apply_ypr(xquat0, yaw=0.0, pitch=pitch, roll=roll)

            # approach & grasp
            for pos in np.linspace(hand0, hover_knob, 1000):
                R.control(pos, xquat_app)
                mujoco.mj_step(R.model, R.data)
                if R.memory:
                    R.memory.trace_step(R.model, R.data, phase=f"approach_{phase_tag}")
                if dwell:
                    time.sleep(dwell)
            for pos in np.linspace(hover_knob, grip_knob, 600):
                R.control(pos, xquat_app)
                mujoco.mj_step(R.model, R.data)
                if R.memory:
                    R.memory.trace_step(R.model, R.data, phase=f"descend_{phase_tag}")
                if dwell:
                    time.sleep(dwell)
            R.gripper(False)
            time.sleep(0.25)

            # closed-loop rotation
            plate_start = R.plate_angle()
            plate_goal = None if seg_angle is None else (plate_start + float(seg_angle))

            # estimate angle tolerance from knob radius (small ~ tighter)
            # use geom radius if available
            gid = mujoco.mj_name2id(R.model, mujoco.mjtObj.mjOBJ_GEOM, knob_name)
            stem_rad = float(R.model.geom_size[gid][0]) if gid >= 0 else 0.010
            angle_tol = max(1e-4, 0.5 * stem_rad / max(radius, 1e-6))  # rad
            step_cap = 4.0 * angle_tol

            # ref vector in plate frame for arc position
            rel0_xy = (knob_pos[:2] - plate_center[:2]).copy()

            last_progress = plate_start
            stalled_iters = 0
            iters = 0

            last_pos_des = R.data.body("panda_hand").xpos.copy()
            last_quat_d = xquat_app.copy()

            while True:
                plate_now = R.plate_angle()
                iters += 1

                # stopping criteria
                if plate_goal is not None:
                    err = R._wrap_to_pi(plate_goal - plate_now)
                    if abs(err) <= angle_tol:
                        # desired angle reached -> stop immediately
                        break
                elif callable(until_done_fn) and (iters % max(1, int(done_check_every)) == 0):
                    if until_done_fn(R.model, R.data):
                        break

                # progress & small lead on EE yaw
                progress = R._wrap_to_pi(plate_now - plate_start)
                yaw_lead = math.copysign(min(abs(progress), 3.0 * angle_tol), progress)
                yaw_target = progress + yaw_lead

                # choose dtheta
                if plate_goal is not None:
                    err = R._wrap_to_pi(plate_goal - plate_now)
                    dtheta = math.copysign(min(abs(err), step_cap), err)
                else:
                    # steady small step if we don't have a numeric goal
                    dtheta = math.copysign(step_cap * 0.8, 1.0)

                theta_cmd = plate_now + dtheta
                phi = math.atan2(rel0_xy[1], rel0_xy[0]) + (theta_cmd - plate_start)
                pos_des = np.array(
                    [
                        plate_center[0] + radius * math.cos(phi),
                        plate_center[1] + radius * math.sin(phi),
                        grip_height,
                    ],
                    dtype=float,
                )

                yaw_rel = R._wrap_to_pi(yaw_target)
                xquat_d = R._apply_ypr(xquat0, yaw=yaw_rel, pitch=pitch, roll=roll)

                for _ in range(int(max(1, arc_inner_iters))):
                    R.control(pos_des, xquat_d)
                    mujoco.mj_step(R.model, R.data)
                    if R.memory:
                        R.memory.trace_step(R.model, R.data, phase=f"rotate_arc_{phase_tag}")
                    if dwell:
                        time.sleep(dwell)

                last_pos_des = pos_des
                last_quat_d = xquat_d

                if abs(plate_now - last_progress) < stall_tol:
                    stalled_iters += 1
                    if stalled_iters > stall_iters:
                        print(f"[spin] segment '{phase_tag}' stalled; breaking early")
                        break
                else:
                    stalled_iters = 0
                    last_progress = plate_now

            # post actions
            if seg_post_action == "hold":
                time.sleep(hold_seconds)
                return {"end_pos": last_pos_des, "end_quat": last_quat_d}
            if seg_post_action == "release_in_place":
                R.gripper(True)
                time.sleep(0.25)
                return {"end_pos": last_pos_des, "end_quat": last_quat_d}

            R.correct_to(last_pos_des, last_quat_d, threshold=1e-4, steps=80)
            R.gripper(True)
            time.sleep(0.25)

            if seg_post_action == "release_and_hover":
                hover_z = grip_height + 0.15
                start_lift = last_pos_des.copy()
                end_lift = last_pos_des.copy()
                end_lift[2] = hover_z
                for pos in np.linspace(start_lift, end_lift, 500):
                    R.control(pos, last_quat_d)
                    mujoco.mj_step(R.model, R.data)
                    if R.memory:
                        R.memory.trace_step(R.model, R.data, phase=f"ascend_{phase_tag}_vertical")
                return {"end_pos": end_lift, "end_quat": last_quat_d}

            if seg_post_action == "release_and_home":
                hover_z = grip_height + 0.15
                start_lift = last_pos_des.copy()
                end_lift = last_pos_des.copy()
                end_lift[2] = hover_z
                for pos in np.linspace(start_lift, end_lift, 400):
                    R.control(pos, last_quat_d)
                    mujoco.mj_step(R.model, R.data)
                    if R.memory:
                        R.memory.trace_step(R.model, R.data, phase=f"ascend_{phase_tag}_vertical")

                # === CHANGED: go to canonical joint-space home (qpos0) ===
                R.reset_joints_to_home(steps=800)

                return {
                    "end_pos": R.data.body("panda_hand").xpos.copy(),
                    "end_quat": R.data.body("panda_hand").xquat.copy(),
                }

            raise ValueError(f"Unknown seg_post_action '{seg_post_action}'")

        # Single-knob execution only
        if angle_rad is None:
            _segment(knob_name, None, "goal", post_action)
            return
        else:
            _segment(knob_name, float(angle_rad), "angle", post_action)
            return
