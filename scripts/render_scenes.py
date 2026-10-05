"""Render the first frame of every LIBERO task (front camera), as the policy sees it.

Usage:
    MUJOCO_GL=egl python render_scenes.py --out renders               # all 130 tasks
    MUJOCO_GL=egl python render_scenes.py --suites libero_object      # one suite
Use MUJOCO_GL=osmesa on a machine without a GPU (slower).
"""
import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]


def render_task(suite, task_id, size, settle_steps):
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size)
    try:
        env.seed(0)
        env.reset()
        env.set_init_state(suite.get_task_init_states(task_id)[0])  # official eval state 0
        obs = None
        for _ in range(settle_steps):  # let objects settle, arm holds still
            obs, _, _, _ = env.step([0.0] * 7)
        # LIBERO renders upside down; flip so it looks like the camera view
        front = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        return task, front, wrist
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="renders")
    ap.add_argument("--suites", nargs="*", default=SUITES)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--settle-steps", type=int, default=5)
    args = ap.parse_args()

    bench = benchmark.get_benchmark_dict()
    index = []
    for name in args.suites:
        suite = bench[name]()
        os.makedirs(os.path.join(args.out, name), exist_ok=True)
        for i in range(suite.n_tasks):
            task, front, wrist = render_task(suite, i, args.size, args.settle_steps)
            stem = os.path.join(args.out, name, f"{i:02d}_{task.name}")
            imageio.imwrite(stem + "_front.jpg", front, quality=88)
            imageio.imwrite(stem + "_wrist.jpg", wrist, quality=88)
            index.append({"suite": name, "id": i, "name": task.name,
                          "language": task.language, "front": stem + "_front.jpg"})
            print(f"[{name} {i:02d}] {task.language}", flush=True)
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
