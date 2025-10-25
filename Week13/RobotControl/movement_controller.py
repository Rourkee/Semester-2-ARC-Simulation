# RobotControl/movement_controller.py
import time
import numpy as np
import mujoco
from grid_utils import grid_to_layer_position

class MovementController:
    """
    Block pick/place routines that use RobotCore.

    Changes vs. previous version:
    - Pre-open gripper before approach.
    - Pick at TOP-OF-BLOCK (uses geom half-height) with a tiny over-travel before closing.
    - Place to top-of-block height with a small clearance.
    - Fail-safe: always open and home in a finally block so the hand never stays closed.
    """

    def __init__(self, robot):
        self.R = robot  # RobotCore

    def execute_block_task_layer(
        self,
        input_coords,
        row: int,
        col: int,
        layer: int,
        *,
        grid_shape=None,
        block_name: str = "",
    ):
        out = grid_to_layer_position(
            self.R.model, self.R.data, row, col, layer, *(grid_shape or (None, None))
        )
        return self.execute_block_task(
            input_coords, out, block_name=block_name, grid_shape=grid_shape
        )

    def _infer_half_height(self, block_name: str, default: float = 0.02) -> float:
        """
        Return half-height of the block's geom (Z half-size for a box), else default.
        """
        if not block_name:
            return float(default)
        gid = mujoco.mj_name2id(self.R.model, mujoco.mjtObj.mjOBJ_GEOM, block_name)
        if gid >= 0 and int(self.R.model.geom_type[gid]) == mujoco.mjtGeom.mjGEOM_BOX:
            # size = [hx, hy, hz] for box => half heights
            if len(self.R.model.geom_size[gid]) >= 3:
                return float(self.R.model.geom_size[gid][2])
        return float(default)

    def execute_block_task(
        self,
        input_coords,
        output_coords,
        *,
        block_name: str = "",
        grid_shape=None,
    ):
        R = self.R
        model, data = R.model, R.data

        # Read current hand pose once
        xpos0 = data.body("panda_hand").xpos.copy()
        xquat0 = data.body("panda_hand").xquat.copy()

        # Infer top-of-block height from the actual geom (more robust than raw target z)
        half = self._infer_half_height(block_name, default=0.02)

        # Make copies so we can adjust Z safely
        inp = np.array(input_coords, float).copy()
        out = np.array(output_coords, float).copy()

        # PICK waypoints: aim at top face, then a tiny over-travel before closing
        pick_touch = inp.copy()
        pick_touch[2] += (half + 0.004)  # just above the top
        pick_down = pick_touch.copy()
        pick_down[2] -= 0.006            # gentle over-travel into contact

        # PLACE waypoints: just above the destination top face (small clearance)
        place_dn = out.copy()
        place_dn[2] += (half + 0.002)

        # Hovers
        hover_in = pick_touch.copy()
        hover_in[2] += 0.15
        hover_out = place_dn.copy()
        hover_out[2] += 0.15

        # Segment plan (mirrors reliable dispenser timings)
        motion = [
            ("preopen",       None),
            ("approach_input", np.linspace(xpos0,     hover_in,  1500)),
            ("descend_input",  np.linspace(hover_in,  pick_touch, 800)),
            ("overtravel_in",  np.linspace(pick_touch, pick_down, 200)),
            ("grasp",          None),
            ("ascend_input",   np.linspace(pick_down, hover_in,   800)),
            ("return_home",    np.linspace(hover_in,  xpos0,     1500)),
            ("approach_output",np.linspace(xpos0,     hover_out, 1500)),
            ("descend_output", np.linspace(hover_out, place_dn,   800)),
            ("release",        None),
            ("ascend_output",  np.linspace(place_dn,  hover_out,  800)),
            ("final_home",     np.linspace(hover_out, xpos0,      800)),
        ]

        try:
            for phase, path in motion:
                if phase == "preopen":
                    # Never dive with a closed gripper
                    R.gripper(True)
                    time.sleep(0.05)
                    continue

                if phase == "grasp":
                    R.gripper(False)
                    time.sleep(0.25)
                    if R.memory:
                        R.memory.log_event(
                            model,
                            data,
                            kind="pick",
                            label=block_name or "(unknown)",
                            grid_shape=grid_shape,
                        )
                    continue

                if phase == "release":
                    R.gripper(True)
                    time.sleep(0.20)
                    if R.memory:
                        R.memory.log_event(
                            model,
                            data,
                            kind="place",
                            label=block_name or "(unknown)",
                            grid_shape=grid_shape,
                        )
                    continue

                if path is not None and len(path) > 0:
                    # Snap to path start if we drifted
                    start = path[0]
                    if np.linalg.norm(data.body("panda_hand").xpos - start) > 1e-3:
                        R.correct_to(start, xquat0, threshold=1e-3, steps=60)
                    # Follow the path
                    for pos in path:
                        R.control(pos, xquat0)
                        mujoco.mj_step(model, data)
                        if R.memory:
                            R.memory.trace_step(model, data, phase=phase)

        finally:
            # Fail-safe: ensure we don't leave the gripper closed on exception
            try:
                R.gripper(True)
                if hasattr(R, "zero_joint_ctrl"):
                    R.zero_joint_ctrl()
                R.reset_joints_to_home()
            except Exception:
                pass
