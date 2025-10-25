# demo.py
import mujoco
import glfw
import numpy as np
import glob
import os
import re
from threading import Thread
import time

from scene_memory import SceneMemory
from dynamicworld import generate_world_xml
from plotting import export_run_pngs

from arc_utils import load_arc_task

# === All symbolic reasoning lives in transformation.py ===
from transformation import (
    apply_rule_to_grid,
    detect_rule_from_pairs,   # train-pair learner (for TEST synthesis)
    detect_program,           # single-source reasoning: geometry + residual edits
    program_to_dsl,           # human-friendly printout
    _maybe_stack_prefix_1,      # x -> 1x  (e.g., 6->16, 7->17, 8->18, ...)
    _maybe_stack_tower_1_to_v,  # x -> 123..x (e.g., 2->12, 3->123, ...)
    apply_stack_prefix_1,
    apply_stack_tower_1_to_v,
)

# 3D adapter (geometry projection & hints; no symbolic reasoning here)
from Ruleto3D import (
    rotation_hint_rad,
    execute_rule,             # project 2D geom rule to per-object (row,col) goals
    collapse_to_cells,        # pick one representative per cell
)

# Grid utils
from grid_utils import (
    apply_input_grid,
    scene_to_grid,
    extract_scene,
    grid_to_positions,
    cache_cell_heights_from_scene,
)

# Robot control
from RobotControl.robot_core import RobotCore
from RobotControl.rotation_controller import RotationController
from RobotControl.movement_controller import MovementController
from RobotControl.bin_dispenser import BinDispenserManager

# GIF recorder
from gif_export import GifRecorder


# -------------------- DSL residual executor --------------------
def _apply_dsl_residuals(grid: np.ndarray, dsl: str) -> np.ndarray:
    """
    Execute residual edits in the DSL, in order.
    Supports:
      - recolor_where(value==X,to=Y)
      - recolor(r,c,src,dst)
      - add(r,c,v)
      - remove(r,c)
    """
    out = np.array(grid, int, copy=True)
    if not dsl:
        return out

    # Extract sequence body if DSL is "seq(...)"
    m = re.match(r"\s*seq\((.*)\)\s*$", dsl)
    body = m.group(1) if m else dsl

    # split into ops and run left->right
    for tok in [t.strip() for t in body.split(";") if t.strip()]:
        # recolor_where(value==X,to=Y)
        m1 = re.match(r"^recolor_where\(value==(\d+),\s*to=(\d+)\)$", tok)
        if m1:
            s = int(m1.group(1)); d = int(m1.group(2))
            out = np.where(out == s, d, out)
            continue

        # recolor(r,c,src,dst)
        m1b = re.match(r"^recolor\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)$", tok)
        if m1b:
            r, c, s, d = map(int, m1b.groups())
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and out[r, c] == s:
                out[r, c] = d
            continue

        # add(r,c,v)
        m2 = re.match(r"^add\((\d+),\s*(\d+),\s*(\d+)\)$", tok)
        if m2:
            r, c, v = map(int, m2.groups())
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                out[r, c] = v
            continue

        # remove(r,c)
        m3 = re.match(r"^remove\((\d+),\s*(\d+)\)$", tok)
        if m3:
            r, c = map(int, m3.groups())
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                out[r, c] = 0
            continue

        # ignore unknown tokens (safe no-op)
    return out


# ---- remap residual coordinates from rotated frame back to identity ----
def _invert_rotate_coords(r, c, H, W, rot_name):
    if rot_name == "rotate_180":
        return (H - 1 - r, W - 1 - c)
    elif rot_name == "rotate_90":      # assuming CW 90 on square grids (H==W)
        return (c, H - 1 - r)
    elif rot_name == "rotate_270":     # assuming CW 270 == CCW 90
        return (W - 1 - c, r)
    else:
        return (r, c)

def _remap_residuals_to_identity(dsl: str, geom_rule: str, grid_shape) -> str:
    """
    If geom_rule is a rotation, rewrite coordinates inside add/remove/recolor(r,c,src,dst)
    into the identity frame so that we can execute residuals without rotating the plate.
    recolor_where(...) is unaffected.
    """
    if not isinstance(geom_rule, str) or not geom_rule.startswith("rotate_"):
        return dsl or ""

    H, W = map(int, grid_shape)
    if not dsl:
        return ""

    m = re.match(r"\s*seq\((.*)\)\s*$", dsl)
    body = m.group(1) if m else dsl
    out_tokens = []

    for tok in [t.strip() for t in body.split(";") if t.strip()]:
        m_add = re.match(r"^add\((\d+),\s*(\d+),\s*(\d+)\)$", tok)
        m_rem = re.match(r"^remove\((\d+),\s*(\d+)\)$", tok)
        m_rec = re.match(r"^recolor\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)$", tok)

        if m_add:
            r, c, v = map(int, m_add.groups())
            ri, ci = _invert_rotate_coords(r, c, H, W, geom_rule)
            out_tokens.append(f"add({ri},{ci},{v})")
        elif m_rem:
            r, c = map(int, m_rem.groups())
            ri, ci = _invert_rotate_coords(r, c, H, W, geom_rule)
            out_tokens.append(f"remove({ri},{ci})")
        elif m_rec:
            r, c, s, d = map(int, m_rec.groups())
            ri, ci = _invert_rotate_coords(r, c, H, W, geom_rule)
            out_tokens.append(f"recolor({ri},{ci},{s},{d})")
        else:
            # Keep recolor_where(...) and other tokens unchanged
            out_tokens.append(tok)

    body2 = ";".join(out_tokens).strip(";")
    return f"seq({body2})" if m else body2


def _apply_program_to_grid(input_grid: np.ndarray, prog) -> np.ndarray:
    """
    Apply the program: first the geometric rule, then residuals parsed from the DSL.
    Falls back to identity geometry if missing.
    """
    geom = getattr(prog, "geom_rule", "identity")
    base = apply_rule_to_grid(np.asarray(input_grid, int), geom if isinstance(geom, str) else "identity")
    dsl = program_to_dsl(prog)
    return _apply_dsl_residuals(base, dsl)


class Demo:
    qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    K = [600.0, 600.0, 600.0, 30.0, 30.0, 30.0]
    height, width = 600, 600

    def __init__(self):
        # Small default world so the viewer can open; each case rebuilds it.
        generate_world_xml((6, 6), write_path="world.xml")
        self.model = mujoco.MjModel.from_xml_path("world.xml")
        self.data = mujoco.MjData(self.model)

        # Camera
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.fixedcamid = -1
        self.cam.lookat = np.array([0.3, 0.0, 0.2])
        self.cam.distance = 1.6
        self.cam.azimuth = 135.0
        self.cam.elevation = -20.0

        self.scene = mujoco.MjvScene(self.model, maxgeom=2000)
        self.memory = SceneMemory(root_dir="scene_memory")
        self.step_done = False
        self.auto_quit_when_done = True
        self.run = True

        # Controllers
        self.robot = RobotCore(self.model, self.data, self.memory, self.K, self.qpos0)
        self.rotate = RotationController(self.robot)
        self.move = MovementController(self.robot)

        # prep: open gripper + home
        self.robot.gripper(True)
        for i in range(1, 8):
            self.data.joint(f"panda_joint{i}").qpos = self.qpos0[i - 1]
        mujoco.mj_forward(self.model, self.data)

        # mouse state
        self._mouse_left = False
        self._mouse_right = False
        self._last_x = 0.0
        self._last_y = 0.0

        # GIF recorder & current run tracking
        self.gifrec = GifRecorder(fps=10, max_seconds=15, max_width=720)
        self._current_run_dir = None

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

        # Learn a single rule from training pairs (legacy geom-only)
        learned_rule = None
        try:
            train_pairs = [(np.asarray(a, int), np.asarray(b, int))
                           for a, b in zip(train_ins, train_outs)]
            if train_pairs:
                learned_rule = detect_rule_from_pairs(train_pairs, allowed=None, prefer_non_identity=True)
                print(f"inferred rule from training pairs: {learned_rule}")
        except Exception as e:
            print(f"train-pair rule inference failed: {e}")
            learned_rule = None

        # Select IOs
        inputs, outputs = (train_ins, train_outs) if mode == "train" else (test_ins, [])

        for ci, input_grid in enumerate(inputs):
            try:
                print(f"=== CASE {ci+1} ===")

                # Expected: given in TRAIN, synthesized for TEST (prefer full program)
                if mode == "train":
                    expected_output = outputs[ci] if ci < len(outputs) else np.zeros_like(input_grid, dtype=int)
                else:
                    # Try parametric STACK rules first (they generalize across colors/positions)
                    expected_output = None
                    try:
                        if train_ins and train_outs:
                            a0 = np.asarray(train_ins[0], int)
                            b0 = np.asarray(train_outs[0], int)

                            if _maybe_stack_prefix_1(a0, b0):
                                expected_output = apply_stack_prefix_1(np.asarray(input_grid, int))
                                print("synthesized expected via rule: stack prefix '1' on every nonzero (x→1x).")
                            elif _maybe_stack_tower_1_to_v(a0, b0):
                                expected_output = apply_stack_tower_1_to_v(np.asarray(input_grid, int))
                                print("synthesized expected via rule: tower [1..v] at each nonzero.")
                    except Exception as e:
                        print(f"stack-rule detection failed: {e}")

                    # Fall back to the existing single-pair program learner
                    if expected_output is None:
                        learned_prog = None
                        try:
                            if train_ins and train_outs:
                                a0 = np.asarray(train_ins[0], int)
                                b0 = np.asarray(train_outs[0], int)
                                learned_prog = detect_program(a0, b0)
                        except Exception as e:
                            print(f"detect_program failed on train[0]: {e}")
                            learned_prog = None

                        if learned_prog is not None:
                            expected_output = _apply_program_to_grid(np.asarray(input_grid, int), learned_prog)
                            print(f"synthesized expected via learned program: {program_to_dsl(learned_prog)}")
                        elif isinstance(learned_rule, str) and learned_rule != "unknown":
                            expected_output = apply_rule_to_grid(np.asarray(input_grid, int), learned_rule)
                            print(f"synthesized expected via learned rule: {learned_rule}")
                        else:
                            expected_output = np.asarray(input_grid, int)
                            print("no learned program or rule; defaulting to identity target.")

                already_saved = False
                rotation_rule_used = None

                # ---- Rebuild world per grid size & values (0-cells are NOT spawned) ----
                H, W = input_grid.shape
                generate_world_xml((H, W), write_path="world.xml", grid_values=input_grid)

                self.model = mujoco.MjModel.from_xml_path("world.xml")
                self.data = mujoco.MjData(self.model)
                self.scene = mujoco.MjvScene(self.model, maxgeom=10000)

                # Rebind controllers
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
                self._current_run_dir = run_dir
                self.memory.trace_step(self.model, self.data, phase="case_start")

                # 1) build scene
                apply_input_grid(self.model, self.data, input_grid)
                mujoco.mj_forward(self.model, self.data)
                self.bin_disp.update()
                self.memory.trace_step(self.model, self.data, phase="post_scene")

                scene = extract_scene(self.model, self.data, *input_grid.shape)
                cache_cell_heights_from_scene(self.model, self.data, *input_grid.shape)
                print("Input Grid:")
                print(np.array(input_grid))

                # -------------------- (A) symbolic reasoning (2D) --------------------
                prog = detect_program(np.asarray(input_grid, int), np.asarray(expected_output, int))
                dsl_full = program_to_dsl(prog)
                print(f"Detected Program: {dsl_full}")

                # Suppress rotation if geom is rotate_* and DSL contains only residual ops
                def _only_residual_ops(dsl: str) -> bool:
                    if not dsl:
                        return True
                    m = re.match(r"\s*seq\((.*)\)\s*$", dsl)
                    body = m.group(1) if m else dsl
                    toks = [t.strip() for t in body.split(";") if t.strip()]
                    # allow only these residual ops
                    for t in toks:
                        if re.match(r"^recolor_where\(value==\d+,\s*to=\d+\)$", t):
                            continue
                        if re.match(r"^recolor\(\d+,\s*\d+,\s*\d+,\s*\d+\)$", t):
                            continue
                        if re.match(r"^add\(\d+,\s*\d+,\s*\d+\)$", t):
                            continue
                        if re.match(r"^remove\(\d+,\s*\d+\)$", t):
                            continue
                        # any other token (e.g., mirror_x, transpose_main, translate_*, etc.)
                        return False
                    return True

                if isinstance(getattr(prog, "geom_rule", "identity"), str) \
                and prog.geom_rule.startswith("rotate_") \
                and _only_residual_ops(dsl_full):
                    prog.geom_rule = "identity"
                    print("rotation suppressed: residual-only program.")

                # (1) Build a rotation-free residual script by inverting residual coords if needed
                dsl_residual_identity = _remap_residuals_to_identity(
                    dsl_full, getattr(prog, "geom_rule", "identity"), input_grid.shape
                )

                # (2) If rotation-free residuals alone hit the expected output, suppress rotation
                try:
                    pred_no_rotate = _apply_dsl_residuals(np.asarray(input_grid, int), dsl_residual_identity)
                    if np.array_equal(pred_no_rotate, np.asarray(expected_output, int)):
                        if hasattr(prog, "geom_rule"):
                            prog.geom_rule = "identity"
                        dsl_full = dsl_residual_identity  # replace for downstream checks/prints
                        print("rotation suppressed by residual remap: identity residuals reach expected.")
                except Exception:
                    pass

                # (3) Final guard: if residuals on original input reproduce target, force identity
                try:
                    alt = _apply_dsl_residuals(np.asarray(input_grid, int), dsl_full)
                    if np.array_equal(alt, np.asarray(expected_output, int)) and getattr(prog, "geom_rule", "identity") != "identity":
                        prog.geom_rule = "identity"
                        print("set geom_rule=identity (residuals alone explain the output).")
                except Exception:
                    pass

                # (B) physical geometry: rotation
                if isinstance(prog.geom_rule, str) and prog.geom_rule.startswith("rotate_") and prog.geom_rule != "identity":
                    print(f"rotate plate: {prog.geom_rule}")
                    ang = rotation_hint_rad(prog.geom_rule)
                    if ang is not None:
                        ang0 = self.robot.plate_angle()
                        self.rotate.spin_plate_knob(
                            angle_rad=ang,
                            post_action="release_hover_then_home",
                            hold_seconds=0.6
                        )
                        ang1 = self.robot.plate_angle()
                        if abs(self.robot._wrap_to_pi((ang0 + float(ang)) - ang1)) <= np.deg2rad(2.0):
                            print("angle target reached.")
                    mujoco.mj_forward(self.model, self.data)
                    self.bin_disp.update()
                    rotation_rule_used = prog.geom_rule

                    # Pre-check after rotation in fixed ARC frame (no inventory yet)
                    scene = extract_scene(self.model, self.data, *input_grid.shape)
                    cache_cell_heights_from_scene(self.model, self.data, *input_grid.shape)
                    final_after_rotate = scene_to_grid(self.model, self.data, *input_grid.shape)
                    final_after_rotate_fixed = apply_rule_to_grid(final_after_rotate, rotation_rule_used)
                    print("Output Grid (after rotation precheck, fixed ARC frame):")
                    print(np.array(final_after_rotate_fixed))
                    if np.array_equal(final_after_rotate_fixed, expected_output):
                        if not already_saved:
                            already_saved = True
                            self.step_done = True
                            folder = self.memory.save_success(
                                self.model, self.data,
                                input_grid=input_grid,
                                expected_output=expected_output,
                                rule=rotation_rule_used,
                                task_path=task_path,
                                split=mode,
                                scene_override=scene,
                            )
                            try:
                                export_run_pngs(folder or run_dir, timestamp=True, show=False)
                            except Exception:
                                pass
                        return

                # Frame to reconcile inventory/moves in (plate frame)
                if rotation_rule_used:
                    inv = {"rotate_90": "rotate_270",
                           "rotate_270": "rotate_90",
                           "rotate_180": "rotate_180"}.get(rotation_rule_used, "identity")
                    expected_in_plate = apply_rule_to_grid(np.asarray(expected_output, int), inv)
                else:
                    expected_in_plate = np.asarray(expected_output, int)

                # (C) physical 3D rules — goal mapping + stash/cycle-breaking moves
                nonrot_geoms = ("mirror_x", "mirror_y", "transpose_main", "transpose_anti")
                is_translate = isinstance(prog.geom_rule, str) and prog.geom_rule.startswith("translate_")
                use_simple_translate = bool(is_translate)
                do_move_planner = (prog.geom_rule in nonrot_geoms) and (prog.geom_rule != "identity")

                # ---------- Simple deterministic translator ----------
                if use_simple_translate:
                    # Parse "translate_dr_dc"
                    try:
                        _, dr_str, dc_str = prog.geom_rule.split("_")
                        dr, dc = int(dr_str), int(dc_str)
                    except Exception:
                        dr = dc = 0

                    print(f"translate: dr={dr}, dc={dc}")
                    scene = extract_scene(self.model, self.data, *input_grid.shape)
                    H2, W2 = input_grid.shape

                    # Collect movable grid blocks
                    movers = []
                    for nm, obj in scene.items():
                        r, c = int(obj.get("row", -1)), int(obj.get("col", -1))
                        if 0 <= r < H2 and 0 <= c < W2 and nm.startswith("G"):
                            movers.append((r, c, nm))

                    # Sort to avoid stepping into targets:
                    # - If dc>0 (right), move rightmost first (col desc). If dc<0 (left), col asc.
                    # - If dr>0 (down), move bottommost first (row desc). If dr<0 (up), row asc.
                    def key(item):
                        r, c, _ = item
                        kc = -c if dc > 0 else (c if dc < 0 else 0)
                        kr = -r if dr > 0 else (r if dr < 0 else 0)
                        return (kc, kr)

                    movers.sort(key=key)

                    # Trash pose for off-grid dumps
                    try:
                        bin_c, _ = self.bin_disp._site_pose(self.bin_disp.trash_site)
                    except Exception:
                        bin_c = np.array([0.0, 0.0, 0.0], float)
                    bin_drop = bin_c.copy(); bin_drop[2] += 0.03

                    # Execute moves
                    for r, c, nm in movers:
                        r2, c2 = r + dr, c + dc
                        src_pos = scene[nm]["position"]
                        if 0 <= r2 < H2 and 0 <= c2 < W2:
                            dst_pos = grid_to_positions(self.model, self.data, r2, c2, H2, W2)
                        else:
                            dst_pos = bin_drop  # off-grid → bin it
                        self.move.execute_block_task(src_pos, dst_pos, block_name=nm, grid_shape=input_grid.shape)
                        mujoco.mj_forward(self.model, self.data)
                        self.bin_disp.update()

                    # Refresh scene after translation
                    scene = extract_scene(self.model, self.data, *input_grid.shape)

                # ---------- Generic move planner for mirrors/transposes ----------
                if do_move_planner:
                    print(f"Plan: block moves for geometry '{prog.geom_rule}'")

                    # Collapse scene: one representative per occupied cell (handles stacks)
                    scene = extract_scene(self.model, self.data, *input_grid.shape)
                    scene_cells = collapse_to_cells(scene)

                    # Compute per-object goals and “to_remove” set (off-grid after mapping)
                    goal = execute_rule(scene_cells, prog.geom_rule, grid_shape=input_grid.shape, strict=False)
                    to_remove = {nm for nm in scene_cells.keys() if nm not in goal}

                    # Spatial tolerances
                    p00 = grid_to_positions(self.model, self.data, 0, 0, *input_grid.shape)
                    p01 = grid_to_positions(self.model, self.data, 0, 1, *input_grid.shape)
                    CELL_PITCH = float(np.linalg.norm(p01[:2] - p00[:2]) or 0.1)
                    EPS_POS = 0.25 * CELL_PITCH

                    moved: set[str] = set()
                    stashed_order: list[str] = []
                    stashed_set: set[str] = set()

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

                    def is_grid_block(nm: str) -> bool:
                        return nm and nm.startswith("G")

                    def is_conflicting(target_pos, ignore=None):
                        """Return the name of a cell-occupying block currently at target_pos (within EPS), else None."""
                        ignore = ignore or set()
                        for nm, obj in scene.items():
                            if nm in ignore:
                                continue
                            if not is_grid_block(nm):
                                continue
                            if nm not in moved and np.linalg.norm(obj["position"] - target_pos) <= EPS_POS:
                                return nm
                        return None

                    # Pre-stash blocks that map off-grid
                    if to_remove:
                        print(f"pre-stash {len(to_remove)} blocks with no goal.")
                        refresh_scene()
                        for nm in list(to_remove):
                            if nm not in scene or not is_grid_block(nm):
                                continue
                            if nm not in stashed_set:
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
                        if not is_grid_block(block_id):
                            return
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
                        if not is_grid_block(name):
                            return True
                        if name in moved:
                            return True
                        if seen is None:
                            seen = set()
                        if name in seen:
                            # Cycle detected → stash this blocker to break the loop
                            print(f"cycle detected on {name} → stash as breaker.")
                            stash_if_needed(name)
                            try_place_stashed()
                            return True
                        seen.add(name)

                        if name not in goal:
                            # No target cell → stash out of the way
                            stash_if_needed(name)
                            try_place_stashed()
                            return True

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

                        # fallback: stash the current name and continue
                        stash_if_needed(name)
                        try_place_stashed()
                        return False

                    if goal:
                        print(f"attempt moves for {len(goal)} blocks.")
                    for name in list(goal.keys()):
                        if not self.run:
                            break
                        move_block(name)

                    # After move planner, refresh expected_in_plate and continue to edits
                    scene = extract_scene(self.model, self.data, *input_grid.shape)

                # (D) Inventory reconcile once against the *correct* frame
                self.bin_disp.update()
                if np.any(np.asarray(expected_in_plate, int) > 9):
                    print("reconcile inventory (stacked).")
                    self.bin_disp.reconcile_inventory_stacked(scene, expected_in_plate, grid_to_positions, input_grid.shape)
                else:
                    print("reconcile inventory.")
                    self.bin_disp.reconcile_inventory(scene, expected_in_plate, grid_to_positions, input_grid.shape)

                # 4) Verify in fixed ARC frame
                scene = extract_scene(self.model, self.data, *input_grid.shape)
                final_after_inv = scene_to_grid(self.model, self.data, *input_grid.shape)
                cmp_grid = apply_rule_to_grid(final_after_inv, rotation_rule_used) if rotation_rule_used else final_after_inv
                print("Output Grid (final, fixed ARC frame):")
                print(np.array(cmp_grid))
                matched = np.array_equal(cmp_grid, expected_output)
                print("Matches Expected Output (final):", matched)

                if matched and not already_saved:
                    already_saved = True
                    self.step_done = True
                    folder = self.memory.save_success(
                        self.model, self.data,
                        input_grid=input_grid,
                        expected_output=expected_output,
                        rule=rotation_rule_used or getattr(prog, "geom_rule", "identity"),
                        task_path=task_path,
                        split=mode,
                        scene_override=scene,
                    )
                    try:
                        export_run_pngs(folder or run_dir, timestamp=True, show=False)
                    except Exception:
                        pass
                    return

                if not matched and not already_saved:
                    already_saved = True
                    folder = self.memory.save_abort(
                        self.model, self.data,
                        input_grid=input_grid,
                        grid_shape=input_grid.shape,
                        reason="final grid != expected",
                    )
                    if folder:
                        print(f"saved trace (abort) → {folder}")

                export_dir = folder or run_dir
                try:
                    outs = export_run_pngs(export_dir, timestamp=True, show=False)
                    print(f"Saved PNGs → {outs}")
                except Exception as e:
                    print(f"Export failed for '{export_dir}': {e}")

            except Exception:
                import traceback
                print("[case] Exception occurred:")
                traceback.print_exc()
                self.step_done = True
                return

    # ---------- Camera helpers ----------
    def _pan_world_xy(self, ndx, ndy):
        """Pan camera parallel to world XY plane."""
        scale = max(0.5, float(self.cam.distance))
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_H, -ndx * scale, 0.0, self.scene, self.cam)
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_V,  0.0,  ndy * scale, self.scene, self.cam)

    def render(self):
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 2)
        window = glfw.create_window(self.width, self.height, "Demo", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)  # vsync

        # seed cursor baseline
        try:
            cx, cy = glfw.get_cursor_pos(window)
            self._last_x, self._last_y = cx, cy
        except Exception:
            pass

        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        # Mouse callbacks
        def _mouse_button_cb(win, button, action, mods):
            if button == glfw.MOUSE_BUTTON_LEFT:
                self._mouse_left = (action == glfw.PRESS)
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self._mouse_right = (action == glfw.PRESS)
            x, y = glfw.get_cursor_pos(win)
            self._last_x, self._last_y = x, y

        def _cursor_pos_cb(win, xpos, ypos):
            if not hasattr(self, "_last_x"):
                self._last_x, self._last_y = xpos, ypos
            dx = xpos - self._last_x
            dy = ypos - self._last_y
            self._last_x, self._last_y = xpos, ypos
            if not (self._mouse_left or self._mouse_right):
                return
            w, h = glfw.get_window_size(win)
            if w <= 0 or h <= 0:
                return
            ndx = dx / max(w, 1)
            ndy = dy / max(h, 1)

            if self._mouse_left:
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_H, ndx, 0.0, self.scene, self.cam)
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, 0.0, ndy, self.scene, self.cam)
            elif self._mouse_right:
                topdown = abs(self.cam.elevation) > 60.0
                force_xy = (glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                            glfw.get_key(win, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
                if topdown or force_xy:
                    self._pan_world_xy(ndx, ndy)
                else:
                    mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_H, ndx, 0.0, self.scene, self.cam)
                    mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_V, 0.0, ndy, self.scene, self.cam)

        def _scroll_cb(win, xoff, yoff):
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoff, self.scene, self.cam)

        glfw.set_mouse_button_callback(window, _mouse_button_cb)
        glfw.set_cursor_pos_callback(window, _cursor_pos_cb)
        glfw.set_scroll_callback(window, _scroll_cb)

        # lazily start the GIF recorder once we know the run_dir
        already_started_gif = False

        while not glfw.window_should_close(window):
            if self.auto_quit_when_done and self.step_done:
                glfw.set_window_should_close(window, True)
                break

            # start recorder when step() creates a run dir
            if not already_started_gif and self._current_run_dir:
                try:
                    self.gifrec.start(self._current_run_dir)
                    already_started_gif = True
                except Exception:
                    pass
            # ---

            w, h = glfw.get_framebuffer_size(window)
            viewport.width = w
            viewport.height = h
            mujoco.mjv_updateScene(self.model, self.data, opt, pert, self.cam,
                                   mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_render(viewport, self.scene, context)

            # capture frame after rendering (before swap)
            try:
                self.gifrec.tick(context, viewport)
            except Exception:
                pass
            # ---

            glfw.swap_buffers(window)
            glfw.poll_events()

        self.run = False

        # encode GIF on exit (if any frames captured)
        try:
            gif_path = self.gifrec.encode_gif()
            if gif_path:
                print(f"[gif] Saved simulation GIF → {gif_path}")
        except Exception:
            pass
        # ---

        glfw.terminate()

    def start(self):
        step_thread = Thread(target=self.step)
        step_thread.start()
        self.render()


if __name__ == "__main__":
    Demo().start()
