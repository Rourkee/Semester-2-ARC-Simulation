##### movement_controller.py
import time
import numpy as np
import mujoco


class MovementController:
    """
    Block pick/place routines that use RobotCore.
    """

    def __init__(self, robot):
        self.R = robot  # RobotCore

    def execute_block_task(self, input_coords, output_coords, *, block_name: str = "", grid_shape=None):
        R = self.R
        xpos0 = R.data.body("panda_hand").xpos.copy()
        xquat0 = R.data.body("panda_hand").xquat.copy()

        hover_input = input_coords.copy();  hover_input[2]  += 0.15
        hover_output = output_coords.copy(); hover_output[2] += 0.15

        motion = [
            ("approach_input", np.linspace(xpos0, hover_input, 2000)),
            ("descend_input",  np.linspace(hover_input, input_coords, 1000)),
            ("grasp",          None),
            ("ascend_input",   np.linspace(input_coords, hover_input, 1000)),
            ("return_home",    np.linspace(hover_input, xpos0, 1000)),
            ("approach_output",np.linspace(xpos0, hover_output, 2000)),
            ("descend_output", np.linspace(hover_output, output_coords, 1000)),
            ("release",        None),
            ("ascend_output",  np.linspace(output_coords, hover_output, 1000)),
            ("final_home",     np.linspace(hover_output, xpos0, 1000)),
        ]

        for phase, path in motion:
            if phase == "grasp":
                R.gripper(False); time.sleep(0.5)
                if R.memory:
                    R.memory.log_event(R.model, R.data, kind="pick",
                                       label=block_name or "(unknown)",
                                       grid_shape=grid_shape)
                # update heights cache if caller wants
            elif phase == "release":
                R.gripper(True); time.sleep(0.5)
                if R.memory:
                    R.memory.log_event(R.model, R.data, kind="place",
                                       label=block_name or "(unknown)",
                                       grid_shape=grid_shape)
            elif path is not None:
                start = path[0]
                if np.linalg.norm(R.data.body("panda_hand").xpos - start) > 1e-3:
                    R.correct_to(start, xquat0, threshold=1e-3, steps=60)
                for pos in path:
                    R.control(pos, xquat0)
                    mujoco.mj_step(R.model, R.data)
                    if R.memory:
                        R.memory.trace_step(R.model, R.data, phase=phase)

        R.reset_joints_to_home()
