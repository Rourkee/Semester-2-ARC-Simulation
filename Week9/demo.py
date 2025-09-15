# Modified by Rourke Young (2025)
# Original code from Panda Robot in MuJoCo (Apache 2.0 License)
##### demo.py
import mujoco
import glfw
import numpy as np
import time
import glob
import os
import re
from threading import Thread

from scene_memory import SceneMemory
from dynamicworld import generate_world_xml

from arc_utils import load_arc_task
from transformation import detect_transformation

from grid_utils import (
    apply_input_grid,
    scene_to_grid,
    extract_scene,
    grid_to_positions,
    cache_cell_heights_from_scene,
)

from Ruleto3D import (
    execute_rule,
    rotation_hint_rad,
)

from plotting import export_run_pngs

# Robot control modules 
from RobotControl.robot_core import RobotCore
from RobotControl.rotation_controller import RotationController
from RobotControl.movement_controller import MovementController
from RobotControl.bin_dispenser import BinDispenserManager


class Demo:
    qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    K = [600.0, 600.0, 600.0, 30.0, 30.0, 30.0]
    height, width = 600, 600
    fps = 40

    def __init__(self):
        # Build a small default world so the viewer can open and each case rebuilds the world later.
        generate_world_xml((3, 3), write_path="world.xml")

        self.model = mujoco.MjModel.from_xml_path("world.xml")
        self.data = mujoco.MjData(self.model)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = 0
        self.scene = mujoco.MjvScene(self.model, maxgeom=2000)
        self.memory = SceneMemory(root_dir="scene_memory")
        self.step_done = False
        self.auto_quit_when_done = True
        self.run = True

        # Compose controllers
        self.robot = RobotCore(self.model, self.data, self.memory, self.K, self.qpos0)
        self.rotate = RotationController(self.robot)
        self.move = MovementController(self.robot)

        # prepare: open gripper and home joints
        self.robot.gripper(True)
        for i in range(1, 8):
            self.data.joint(f"panda_joint{i}").qpos = self.qpos0[i - 1]
        mujoco.mj_forward(self.model, self.data)
        self.bin_disp = BinDispenserManager(self.model, self.data, self.robot, spawn_delay=0.4)
        self.bin_disp._spawn_block_into_dispenser()  # force one block into the opening
        self.bin_disp.update()


    # ---------------- main pipeline ----------------
    def step(self):
        available_tasks = glob.glob("arc_tasks/*.json")
        print("Available ARC task files:")
        for i, fname in enumerate(available_tasks):
            print(f"{i}: {os.path.basename(fname)}")
        choice = int(input("Select task index: "))
        task_path = available_tasks[choice]

        mode = input("Run mode (train/test): ").strip().lower()
        train_ins, train_outs, test_ins, test_outs = load_arc_task(task_path)
        inputs, outputs = (train_ins, train_outs) if mode == "train" else (test_ins, test_outs)

        for i, input_grid in enumerate(inputs):
            try:
                print(f"=== CASE {i+1} ===")
                expected_output = outputs[i] if i < len(outputs) else []

                # ---- Rebuild world per grid size & values (0-cells are NOT spawned) ----
                H, W = input_grid.shape
                generate_world_xml((H, W), write_path="world.xml", grid_values=input_grid)

                # Rebuild model/data/scene from the freshly generated XML
                self.model = mujoco.MjModel.from_xml_path("world.xml")
                self.data = mujoco.MjData(self.model)
                self.scene = mujoco.MjvScene(self.model, maxgeom=10000)

                # Recreate controllers so they point at the new model/data
                self.robot = RobotCore(self.model, self.data, self.memory, self.K, self.qpos0)
                self.rotate = RotationController(self.robot)
                self.move = MovementController(self.robot)

                # Prep robot state
                self.robot.gripper(True)
                for j in range(1, 8):
                    self.data.joint(f"panda_joint{j}").qpos = self.qpos0[j - 1]
                mujoco.mj_forward(self.model, self.data)
                self.bin_disp = BinDispenserManager(self.model, self.data, self.robot, spawn_delay=0.4)
                self.bin_disp._spawn_block_into_dispenser()
                self.bin_disp.update()

                run_dir = self.memory.trace_start()
                self.memory.trace_step(self.model, self.data, phase="case_start")

                # 1) build scene (color/place as per grid; zeros are hidden/parked if present)
                apply_input_grid(self.model, self.data, input_grid)
                mujoco.mj_forward(self.model, self.data)
                self.bin_disp.update()
                self.memory.trace_step(self.model, self.data, phase="post_scene")

                scene = extract_scene(self.model, self.data, *input_grid.shape)
                cache_cell_heights_from_scene(self.model, self.data, *input_grid.shape)
                print("Input Grid:")
                print(np.array(input_grid))

                # 2) detect rule
                rule = detect_transformation(input_grid, expected_output)
                print(f"Detected Rule: {rule}")
                # --- Inventory-delta fallback: when rule is unknown OR counts differ -> use the dispenser/bin ---
                cur_grid = scene_to_grid(self.model, self.data, *input_grid.shape)
                if (rule == "unknown") or (np.count_nonzero(cur_grid) != np.count_nonzero(expected_output)):
                    # Make sure a block is available and the dispenser state is fresh
                    self.bin_disp.update()
                    self.bin_disp.reconcile_inventory(scene, expected_output, grid_to_positions, input_grid.shape)

                    # refresh & verify
                    scene = extract_scene(self.model, self.data, *input_grid.shape)
                    final_after_inv = scene_to_grid(self.model, self.data, *input_grid.shape)
                    print("Output Grid (after inventory-delta):")
                    print(np.array(final_after_inv))
                    if np.array_equal(final_after_inv, expected_output):
                        self.step_done = True
                        folder = self.memory.save_success(
                            self.model, self.data,
                            input_grid=input_grid,
                            expected_output=expected_output,
                            rule=rule,
                            task_path=task_path,
                            split=mode,
                            scene_override=scene,
                        )
                        try:
                            export_run_pngs(folder or run_dir, timestamp=True, show=False)
                        except Exception:
                            pass
                        return


                # 3) Plate-level rotation (single angle-based spin with angle success check)
                def _is_rotate_rule(r):
                    return isinstance(r, str) and r.strip().lower().startswith("rotate_")

                if _is_rotate_rule(rule):
                    delta_hint = rotation_hint_rad(rule)
                    if delta_hint is not None:
                        ang0 = self.robot.plate_angle()
                        self.rotate.spin_plate_knob(
                            angle_rad=delta_hint,
                            post_action="release_hover_then_home",
                            hold_seconds=0.6
                        )
                        ang1 = self.robot.plate_angle()
                        ang_err = abs(self.robot._wrap_to_pi((ang0 + float(delta_hint)) - ang1))
                        if ang_err <= np.deg2rad(2.0):
                            self.step_done = True
                            folder = self.memory.save_success(
                                self.model, self.data,
                                input_grid=input_grid,
                                expected_output=expected_output,
                                rule=rule,
                                task_path=task_path,
                                split=mode,
                            )
                            print("[rotate] angle target reached.")
                            if folder:
                                print(f"[scene_memory] saved snapshot → {folder}")
                            try:
                                export_run_pngs(folder or run_dir, timestamp=True, show=False)
                            except Exception:
                                pass
                            return

                    mujoco.mj_forward(self.model, self.data)
                    self.bin_disp.update()

                    scene = extract_scene(self.model, self.data, *input_grid.shape)
                    cache_cell_heights_from_scene(self.model, self.data, *input_grid.shape)
                    self.memory.trace_step(self.model, self.data, phase="post_rotate_precheck")

                    final_after_rotate = scene_to_grid(self.model, self.data, *input_grid.shape)
                    print("Output Grid (after rotation precheck):")
                    print(np.array(final_after_rotate))
                    solved = np.array_equal(final_after_rotate, expected_output)
                    print("Matches Expected Output (after rotation only):", solved)

                    if solved:
                        self.step_done = True
                        folder = self.memory.save_success(
                            self.model, self.data,
                            input_grid=input_grid,
                            expected_output=expected_output,
                            rule=rule,
                            task_path=task_path,
                            split=mode,
                            scene_override=scene,
                        )
                        try:
                            export_run_pngs(folder or run_dir, timestamp=True, show=False)
                        except Exception:
                            pass
                        return

                # 4) Compute block-level goal (helpful for off-grid removals)
                def _parse_translate(rule_str: str):
                    s = (rule_str or "").strip().lower()
                    if not s.startswith("translate"):
                        return None
                    m = re.search(r"translate(?:_|\()?\s*(-?\d+)[,_]\s*(-?\d+)\)?", s)
                    return (int(m.group(1)), int(m.group(2))) if m else None

                tx = _parse_translate(rule)
                goal: dict[str, dict] = {}
                to_remove: set[str] = set()

                if tx is not None:
                    dr, dc = tx
                    for name, obj in scene.items():
                        r, c = obj["row"], obj["col"]
                        if r < 0 or c < 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            goal[name] = {"row": int(nr), "col": int(nc)}
                        else:
                            to_remove.add(name)
                else:
                    goal = execute_rule(scene, rule, grid_shape=input_grid.shape)
                    for name in scene.keys():
                        if name not in goal:
                            to_remove.add(name)

                # Spatial tolerances
                p00 = grid_to_positions(self.model, self.data, 0, 0, *input_grid.shape)
                p01 = grid_to_positions(self.model, self.data, 0, 1, *input_grid.shape)
                CELL_PITCH = float(np.linalg.norm(p01[:2] - p00[:2]) or 0.1)
                EPS_POS = 0.25 * CELL_PITCH

                moved: set[str] = set()
                stashed_order: list[str] = []
                stashed_set: set[str] = set()

                # Stash area (out of the grid workspace)
                z_guess = grid_to_positions(self.model, self.data, 0, 0, *input_grid.shape)[2]
                stash_area_origin = np.array([0.3, 0.4, z_guess + 0.08], dtype=float)
                stash_spacing = 0.1

                def refresh_scene():
                    nonlocal scene
                    scene = extract_scene(self.model, self.data, *input_grid.shape)
                    cache_cell_heights_from_scene(self.model, self.data, *input_grid.shape)

                def at_pose(name, pos, eps=EPS_POS):
                    return (name in scene) and (np.linalg.norm(scene[name]["position"] - pos) <= eps)

                def get_stash_position():
                    idx = len(stashed_order)
                    return stash_area_origin + np.array([idx * stash_spacing, 0, 0])

                def is_conflicting(target_pos, ignore=None):
                    ignore = ignore or set()
                    for nm, obj in scene.items():
                        if nm in ignore:
                            continue
                        if nm not in moved and np.linalg.norm(obj["position"] - target_pos) <= EPS_POS:
                            return nm
                    return None

                # Pre-stash everything that has no goal
                if to_remove:
                    refresh_scene()
                    for nm in list(to_remove):
                        if nm in scene and nm not in stashed_set:
                            stash_pos = get_stash_position()
                            self.move.execute_block_task(scene[nm]["position"], stash_pos,
                                                         block_name=nm, grid_shape=input_grid.shape)
                            refresh_scene()
                            stashed_order.append(nm)
                            stashed_set.add(nm)
                            moved.add(nm)

                def try_place_stashed():
                    progress = True
                    while progress:
                        progress = False
                        refresh_scene()
                        for nm in list(stashed_order):
                            if nm not in goal:
                                continue
                            tgt = goal[nm]
                            tgt_pos = grid_to_positions(self.model, self.data, tgt["row"], tgt["col"], *input_grid.shape)
                            if at_pose(nm, tgt_pos):
                                moved.add(nm)
                                stashed_order.remove(nm)
                                stashed_set.remove(nm)
                                progress = True
                                continue
                            blocker = is_conflicting(tgt_pos, ignore={nm})
                            if blocker is None:
                                self.move.execute_block_task(scene[nm]["position"], tgt_pos,
                                                             block_name=nm, grid_shape=input_grid.shape)
                                refresh_scene()
                                moved.add(nm)
                                stashed_order.remove(nm)
                                stashed_set.remove(nm)
                                progress = True

                def stash_if_needed(block_id):
                    if block_id in stashed_set:
                        return
                    if block_id not in scene:
                        return
                    stash_pos = get_stash_position()
                    self.move.execute_block_task(scene[block_id]["position"], stash_pos,
                                                 block_name=block_id, grid_shape=input_grid.shape)
                    refresh_scene()
                    stashed_order.append(block_id)
                    stashed_set.add(block_id)
                    moved.add(block_id)

                def move_block(name, seen=None):
                    # If a blocker or stray id is requested with no goal, stash it safely.
                    if name not in goal:
                        stash_if_needed(name)
                        try_place_stashed()
                        return True

                    if name in moved:
                        return True
                    if seen is None:
                        seen = set()
                    if name in seen:
                        return False
                    seen.add(name)

                    refresh_scene()
                    tgt = goal[name]
                    tgt_pos = grid_to_positions(self.model, self.data, tgt["row"], tgt["col"], *input_grid.shape)

                    if at_pose(name, tgt_pos):
                        moved.add(name)
                        try_place_stashed()
                        return True

                    blocking = is_conflicting(tgt_pos, ignore={name})
                    if blocking and blocking != name:
                        if blocking not in goal:
                            stash_if_needed(blocking)
                        else:
                            if move_block(blocking, seen=seen.copy()):
                                refresh_scene()
                                blocking = is_conflicting(tgt_pos, ignore={name})

                        if blocking and blocking != name and blocking not in stashed_set:
                            stash_if_needed(blocking)

                    refresh_scene()
                    blocking = is_conflicting(tgt_pos, ignore={name})

                    if blocking == name and at_pose(name, tgt_pos, eps=EPS_POS * 1.5):
                        moved.add(name)
                        try_place_stashed()
                        return True

                    if not blocking:
                        if at_pose(name, tgt_pos):
                            moved.add(name)
                            try_place_stashed()
                            return True
                        self.move.execute_block_task(scene[name]["position"], tgt_pos,
                                                     block_name=name, grid_shape=input_grid.shape)
                        refresh_scene()
                        moved.add(name)
                        try_place_stashed()
                        return True

                    stash_if_needed(name)
                    try_place_stashed()
                    return False

                # Initiate moves ONLY for blocks that actually have goals
                for name in list(goal.keys()):
                    if not self.run:
                        break
                    move_block(name)

                final_grid = scene_to_grid(self.model, self.data, *input_grid.shape)
                print("Output Grid:")
                print(np.array(final_grid))
                ok = np.array_equal(final_grid, expected_output)
                print("Matches Expected Output:", ok)
                self.step_done = True

                if ok:
                    folder = self.memory.save_success(
                        self.model, self.data,
                        input_grid=input_grid,
                        expected_output=expected_output,
                        rule=rule,
                        task_path=task_path,
                        split=mode,
                        scene_override=scene,
                    )
                    print(f"[scene_memory] saved snapshot → {folder}")
                else:
                    folder = self.memory.save_abort(
                        self.model, self.data,
                        input_grid=input_grid,
                        grid_shape=input_grid.shape,
                        reason="final grid != expected",
                    )
                    if folder:
                        print(f"[scene_memory] saved trace (abort) → {folder}")

                export_dir = folder or run_dir
                try:
                    outs = export_run_pngs(export_dir, timestamp=True, show=False)
                    print(f"Saved PNGs → {outs}")
                except Exception as e:
                    print(f"Export failed for '{export_dir}': {e}")

            except Exception as e:
                import traceback
                print("[case] Exception occurred:")
                traceback.print_exc()
                self.step_done = True
                return

    def render(self):
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 2)
        window = glfw.create_window(self.width, self.height, "Demo", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)  # vsync on; smooth and avoids busy-wait

        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        while not glfw.window_should_close(window):
            if self.auto_quit_when_done and self.step_done:
                glfw.set_window_should_close(window, True)
                break

            w, h = glfw.get_framebuffer_size(window)
            viewport.width = w
            viewport.height = h
            mujoco.mjv_updateScene(self.model, self.data, opt, pert, self.cam,
                                mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_render(viewport, self.scene, context)
            glfw.swap_buffers(window)
            glfw.poll_events()


        self.run = False
        glfw.terminate()

    def start(self):
        step_thread = Thread(target=self.step)
        step_thread.start()
        self.render()


if __name__ == "__main__":
    Demo().start()

