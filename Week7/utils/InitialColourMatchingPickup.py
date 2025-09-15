"""Demonstrates the Franka Emika Robot System model for MuJoCo."""

import time                          # Provies sleep functions for delays              
from threading import Thread         # Enables parrallel execution for simulation and rendering

import glfw                          # Renders simulation
import mujoco                        # Mojuco physics 
import numpy as np                   # Numerical computation

from MatchingColour import find_colour_match

from BlockPositions import get_input_output_blocks             # Module that Gets block positions 
from AssignColours import AssignColours                     # Module that Randomises block colours -> One I and O matches coloour.
#from IKNewtonRaphson import synthetic_ik, get_pose, get_jacobian, set_qpos        # Module that uses Inverse Kinematic


class Demo:
### Initial parameters of robot ###
    qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]                          # Joint positions
    K = [600.0, 600.0, 600.0, 30.0, 30.0, 30.0]                              # Stiffness value of joints
    height, width = 960, 1280                                                # Size of the window when you run code
    fps = 20  # Rendering framerate.



### Initialising Robot Simulation in Mojuco ###
    def __init__(self) -> None:                                               # Sets up simulation
        self.model = mujoco.MjModel.from_xml_path("world.xml")                # Loads Mujoco simulation of world.xml
        self.data = mujoco.MjData(self.model)                                 # Simulation created based on data (MJdata) of the world.xml etc joints, forces
        self.cam = mujoco.MjvCamera()                                         # Creates camera for simulation. Determines viewpoints in simulation
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED                       # Fixed meaning doesn't move throughout simulation
        self.cam.fixedcamid = 0                                               # Selects first fixed camera from xml file
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)               # Renders the model. Max of 10000 objects.
        self.run = True                                                       # Runs simulation. If set to false simulation stops
        self.gripper(True)                                                    # Gripper opened initially when (true). Closed when (false)
        for i in range(1, 8):                                                 ## Sets initial joint positions. Loops through seven joints (1-7). 
            self.data.joint(f"panda_joint{i}").qpos = self.qpos0[i - 1]       
        mujoco.mj_forward(self.model, self.data)                              # Applied forward kinematics to update simulation


        self.input_blocks, self.output_blocks = get_input_output_blocks(self.model, self.data)                 # Gets block positions
        AssignColours(self.model, self.data, self.input_blocks, self.output_blocks)                            # Assign random colours

        for block in list(self.input_blocks) + list(self.output_blocks):
            print(f"{block}: {self.data.geom(block).xpos}")



    def gripper(self, open=True):                                                         # Controls gripper. Value 0.04 moves fingers apart, 0 closes them.
        self.data.actuator("pos_panda_finger_joint1").ctrl = (0.04, 0)[not open]
        self.data.actuator("pos_panda_finger_joint2").ctrl = (0.04, 0)[not open]

    def control(self, xpos_d, xquat_d):                                                          ## Moves robot arm to desired position (xpos_d) and orientation (xquat_d)
        xpos = self.data.body("panda_hand").xpos                                                 
        xquat = self.data.body("panda_hand").xquat                                                
        jacp = np.zeros((3, self.model.nv))                                                       # Initalise positional Jacobian
        jacr = np.zeros((3, self.model.nv))                                                       # Initalise rotational Jacobian
        bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")           
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)                              # Compute jacobians

        error = np.zeros(6)                                                            ## Position and orientation error
        error[:3] = xpos_d - xpos
        res = np.zeros(3)
        mujoco.mju_subQuat(res, xquat, xquat_d)
        mujoco.mju_rotVecQuat(res, res, xquat)
        error[3:] = -res

        J = np.concatenate((jacp, jacr))                                             # Combines jacobians 
        v = J @ self.data.qvel                                                       # Compute joint velocity
        for i in range(1, 8):
            dofadr = self.model.joint(f"panda_joint{i}").dofadr
            self.data.actuator(f"panda_joint{i}").ctrl = self.data.joint(       # .ctrl -> Control input (torque or force) to joint
                f"panda_joint{i}"
            ).qfrc_bias                                                       # Compensates for gravity/coriolis forces
            self.data.actuator(f"panda_joint{i}").ctrl += (                  # .ctrl -> Control input (torque or force) to joint
                J[:, dofadr].T @ np.diag(self.K) @ error                     # Position/Orientation error between desied and current end-effector pose
            )                                                                # Self.k is stiffness matrix. Converts gain to diagonal matrix. Then selects jacboian correpsonding to current joint
            self.data.actuator(f"panda_joint{i}").ctrl -= (                  
                J[:, dofadr].T @ np.diag(2 * np.sqrt(self.K)) @ v
            )

### Step function to move arm ###
    def step(self) -> None:    
                                                 
        xpos0 = self.data.body("panda_hand").xpos.copy()        # Initial Position of end effector 
        xpos_d = xpos0                                          # Sets desired position to be the same as the current position
        xquat0 = self.data.body("panda_hand").xquat.copy()      # Initial orientiation (Quaternoin) of End Effector
        

        input_blocks, output_blocks = get_input_output_blocks(self.model, self.data)
        matching_blocks = find_colour_match(input_blocks, output_blocks, self.model, self.data)

        print("Matching Blocks:")
        for match in matching_blocks:
            print(f"Input Block: {match[0]} -> Output Block: {match[1]}")

            Input_co_ordinates = input_blocks[match[0]]
            Output_co_ordinates = output_blocks[match[1]]

            diff = np.array(Input_co_ordinates) - np.array(xpos0)   # diff[0] = x, diff[1] = y, diff[2] = z
            
            

            x_steps = list(np.linspace(xpos0[0], Input_co_ordinates[0], 2000))      # Initial position to x,y,z
            y_steps = list(np.linspace(xpos0[1], Input_co_ordinates[1], 2000))
            z_steps = list(np.linspace(xpos0[2], Input_co_ordinates[2], 2000))

            movement_plan = list(zip(x_steps, y_steps, z_steps))

            for step in movement_plan:
                xpos_d = list(step)
              
                





                self.control(xpos_d, xquat0)                                   # Moves robot's joints to desired positions                       
                mujoco.mj_step(self.model, self.data)                          # Updates physics simulation for next step
                time.sleep(1e-3)                                               # Adds small delay to control execution speed

                




### To edit GLFW window (simulation that pops up) ###
    def render(self) -> None:
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 8)
        window = glfw.create_window(self.width, self.height, "Demo", None, None)
        glfw.make_context_current(window)
        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_100
        )
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        while not glfw.window_should_close(window):
            w, h = glfw.get_framebuffer_size(window)
            viewport.width = w
            viewport.height = h
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                opt,
                pert,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene,
            )
            mujoco.mjr_render(viewport, self.scene, self.context)
            time.sleep(1.0 / self.fps)
            glfw.swap_buffers(window)
            glfw.poll_events()
        self.run = False
        glfw.terminate()

    def start(self) -> None:
        step_thread = Thread(target=self.step)
        step_thread.start()
        self.render()


if __name__ == "__main__":8914
Demo().start()