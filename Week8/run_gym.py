# run_gym.py
import time
import traceback
from arc_world_env import ArcWorldEnv  # used directly, so Pylance is happy

def main():
    env = ArcWorldEnv(
        task_file=None,          # auto-pick first task in arc_tasks/
        split="train",
        case_index=0,
        auto_advance_cases=False,
        render_mode="human",
        auto_quit_when_done=False,  # keep viewer open after success
    )

    obs, info = env.reset()
    print("Reset:", info)
    print("obs shape:", getattr(env.observation_space, "shape", None), "->", getattr(obs, "shape", None))

    done = False
    steps = 0
    try:
        while not done and steps < 10:
            print(f"[driver] calling env.step() #{steps+1} ...")
            try:
                # Action ignored in scripted mode; kept for future RL wiring
                action = None
                obs, reward, terminated, truncated, step_info = env.step(action)
            except Exception:
                print("[driver] exception during env.step():")
                traceback.print_exc()
                break

            steps += 1
            done = terminated or truncated
            print(f"[driver] step {steps} -> reward={reward} term={terminated} trunc={truncated} info={step_info}")
            time.sleep(1/20)  # ~20 Hz so you can watch
        input("Episode finished. Press Enter to close the viewer...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
