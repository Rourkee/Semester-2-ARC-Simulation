# arc_gym_env.py
from __future__ import annotations
import os, glob, numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# --- your existing modules (unchanged) ---
from dynamicworld import generate_world_xml
from arc_utils import load_arc_task
from transformation import detect_transformation
from grid_utils import (
    apply_input_grid, scene_to_grid, extract_scene,
    cache_cell_heights_from_scene, grid_to_positions,
)
from Ruleto3D import execute_rule, rotation_hint_rad
from scene_memory import SceneMemory
from plotting import export_run_pngs

from RobotControl.robot_core import RobotCore
from RobotControl.rotation_controller import RotationController
from RobotControl.movement_controller import MovementController


class ArcEpisodeEnv(gym.Env):
    """
    Minimal Gym wrapper around your existing demo:
      - reset(): runs one full scripted episode to completion (no per-step control).
      - step(): no-op; episode already finished (terminated=True).
      - Observation: final grid (6x6 int map, padded).
      - Reward: 1.0 if final grid == expected, else 0.0.
      - Info: task name, case index, grid shape, success flag, rule (if any).
    """
    metadata = {"render_modes": ["none"]}

    def __init__(
        self,
        *,
        tasks_dir: str = "arc_tasks",
        split: str = "train",           # "train" or "test"
        case_index: int | None = None,  # None -> random each episode
        qpos0=(0, -0.785, 0, -2.356, 0, 1.571, 0.785),
        K=(600.0, 600.0, 600.0, 30.0, 30.0, 30.0),
        save_pngs: bool = True,
        render_live: bool = False,      # set True to open your existing viewer
        seed: int | None = None,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        self.tasks = sorted(glob.glob(os.path.join(tasks_dir, "*.json")))
        if not self.tasks:
            raise FileNotFoundError(f"No ARC tasks found in {tasks_dir}")

        self.split = split
        self.case_index_cfg = case_index
        self.qpos0 = np.array(qpos0, dtype=float)
        self.K = np.array(K, dtype=float)
        self.save_pngs = bool(save_pngs)
        self.render_live = bool(render_live)

        # Fixed-shape observation: final grid indices 0..9, padded to 6x6
        self.observation_space = spaces.Box(low=0, high=9, shape=(6, 6), dtype=np.int32)
        # Dummy action: we don't actually act—just keep Gym happy
        self.action_space = spaces.Discrete(1)

        # Episode cache
        self._last_success = False
        self._last_info = {}

    # ---------- Gym API ----------
    def seed(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed(seed)

        # --- choose task & case (non-interactive replacement for input()) ---
        task_path = self._choose_task()
        train_in, train_out, test_in, test_out = load_arc_task(task_path)
        inputs, outputs = (train_in, train_out) if self.split == "train" else (test_in, test_out)
        if not inputs:
            raise ValueError(f"No '{self.split}' cases in {os.path.basename(task_path)}")

        idx = self._choose_case(len(inputs))
        input_grid = inputs[idx]
        expected = outputs[idx] if idx < len(outputs) else None

        H, W = int(input_grid.shape[0]), int(input_grid.shape[1])

        # --- rebuild world exactly like your demo ---
        generate_world_xml((H, W), write_path="world.xml", grid_values=input_grid.tolist())
        model = mujoco.MjModel.from_xml_path("world.xml")
        data = mujoco.MjData(model)

        # Controllers + memory
        memory = SceneMemory(root_dir="scene_memory")
        R = RobotCore(model, data, memory, self.K, self.qpos0)
        rotate = RotationController(R)
        move = MovementController(R)

        # Prep: open gripper + home
        R.gripper(True)
        R.reset_joints_to_home(steps=300)
        mujoco.mj_forward(model, data)

        # Apply input & cache
        apply_input_grid(model, data, input_grid)
        mujoco.mj_forward(model, data)
        cache_cell_heights_from_scene(model, data, H, W)
        scene = extract_scene(model, data, H, W)

        rule = None
        success = False

        # --- your fast rotation path ---
        if expected is not None:
            rule = detect_transformation(input_grid, expected)
            if isinstance(rule, str) and rule.strip().lower().startswith("rotate_"):
                def until_done_fn(m, d):
                    cur = scene_to_grid(m, d, H, W)
                    return np.array_equal(cur, expected)

                _ = rotation_hint_rad(rule)  # optional logging/info
                rotate.spin_plate_knob(angle_rad=None, until_done_fn=until_done_fn, post_action="release_and_home")
                mujoco.mj_forward(model, data)

                # belt-and-braces home like your demo
                R.reset_joints_to_home(steps=800)
                mujoco.mj_forward(model, data)

                if np.array_equal(scene_to_grid(model, data, H, W), expected):
                    success = True

        # --- otherwise: compute goals and do pick-place like your demo ---
        if (expected is not None) and not success:
            goal = execute_rule(scene, rule, grid_shape=(H, W)) if rule else {}
            # move each goal block; your conflict/stash logic is heavy—here we do direct moves,
            # because world was generated per-input and most ARC rules map 1:1 cleanly
            for name, tgt in goal.items():
                src = scene[name]["position"]
                tgt_pos = grid_to_positions(model, data, tgt["row"], tgt["col"], H, W)
                move.execute_block_task(src, tgt_pos, block_name=name, grid_shape=(H, W))
                mujoco.mj_forward(model, data)
                scene = extract_scene(model, data, H, W)

            success = np.array_equal(scene_to_grid(model, data, H, W), expected)

        # --- finalize outputs exactly like your demo ---
        final_grid = scene_to_grid(model, data, H, W)
        info = {
            "task": os.path.basename(task_path),
            "case_index": idx,
            "grid_shape": (H, W),
            "success": bool(success),
            "rule": rule,
        }

        # logging & pngs (same directories you already use)
        memory.log_meta(info)
        if self.save_pngs:
            try:
                export_run_pngs(memory.out_dir, model, data)
            except Exception:
                pass

        # Return obs + info (Gymnasium reset signature)
        obs = self._pad6(final_grid)
        self._last_success, self._last_info = bool(success), info
        return obs, info

    def step(self, action):
        # No per-step control; episode already completed in reset()
        obs = np.zeros((6, 6), dtype=np.int32)
        reward = 1.0 if self._last_success else 0.0
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, dict(self._last_info)

    # ---------- helpers ----------
    def _choose_task(self) -> str:
        return self.tasks[self.rng.integers(0, len(self.tasks))]

    def _choose_case(self, n: int) -> int:
        if self.case_index_cfg is None:
            return int(self.rng.integers(0, n))
        return int(np.clip(self.case_index_cfg, 0, n - 1))

    def _pad6(self, g: np.ndarray) -> np.ndarray:
        out = np.zeros((6, 6), dtype=np.int32)
        h, w = g.shape[:2]
        out[:h, :w] = g
        return out
