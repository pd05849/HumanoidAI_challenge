"""Replay one human teleop demo of LIBERO-90 task 21 ("turn on the stove and put the frying pan on it").

Two replays of the same demo, each saved as a video:
  1. state playback  - sets the simulator to every recorded state (exactly what the human did)
  2. action replay   - starts from the first state and re-sends the recorded 7-number actions
                       (this is how the demos get re-recorded at 256 px for training later)
Both print whether LIBERO's goal check says the task was completed.

Usage:  MUJOCO_GL=egl python replay_demo.py --demo 0
"""
import argparse
import os

import h5py
import imageio.v2 as imageio
import numpy as np
from huggingface_hub import hf_hub_download
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

REPO = "yifengzhu-hf/LIBERO-datasets"
DEMO_FILE = "libero_90/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it_demo.hdf5"
SUITE, TASK_ID, SIZE = "libero_90", 21, 256


def save_video(frames, path, fps=20):
    try:
        imageio.mimsave(path, frames, fps=fps)  # mp4 via imageio-ffmpeg / pyav if installed
    except Exception:
        path = os.path.splitext(path)[0] + ".gif"
        imageio.mimsave(path, frames[::2], duration=2 / fps)  # fallback: gif at half rate
    print("  saved", path)


def frame(obs):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  # LIBERO renders upside down


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--file", default=None, help="local .hdf5 (default: download from Hugging Face)")
    args = ap.parse_args()

    path = args.file or hf_hub_download(REPO, DEMO_FILE, repo_type="dataset")
    with h5py.File(path) as h:
        demo = h["data"][f"demo_{args.demo}"]
        states, actions = demo["states"][()], demo["actions"][()]
        print(f"demo_{args.demo}: {len(actions)} steps (~{len(actions) / 20:.1f} s at 20 Hz), "
              f"obs keys: {list(demo['obs'].keys())}")

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(TASK_ID)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=SIZE, camera_widths=SIZE)
    env.seed(0)
    env.reset()
    print("task:", task.language)

    # 1. state playback: exactly the recorded motion, re-rendered at 256 px
    frames = [frame(env.regenerate_obs_from_state(s)) for s in states]
    print(f"1) state playback   -> task completed: {env.check_success()}")
    save_video(frames, f"demo{args.demo}_state_playback.mp4")

    # 2. action replay: same start, re-send the human's actions through the controller
    obs = env.set_init_state(states[0])
    frames, done = [frame(obs)], False
    for a in actions:
        obs, _, done, _ = env.step(a.tolist())
        frames.append(frame(obs))
    print(f"2) action replay    -> task completed: {env.check_success()}")
    save_video(frames, f"demo{args.demo}_action_replay.mp4")
    env.close()


if __name__ == "__main__":
    main()
