"""Evaluate the push-knob scripted controller on LIBERO-90 task 21.

usage: python sim/run_controller_eval.py [--states 0-9] [--cam 32] [--nocam] [--video DIR] [-v]
"""
import argparse, os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import kitchen_controller as C

TASK_ID = 21


def parse_states(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def make_env(cam, use_cam):
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(TASK_ID)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=cam, camera_widths=cam,
                             use_camera_obs=use_cam)
    env.seed(0)
    env.reset()
    return env, suite.get_task_init_states(TASK_ID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="0-9")
    ap.add_argument("--cam", type=int, default=32)
    ap.add_argument("--nocam", action="store_true")
    ap.add_argument("--video", default=None, help="dir to save mp4s (needs camera)")
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--settle", type=int, default=5)
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_stop", action="store_true", help="keep running after success")
    args = ap.parse_args()

    env, inits = make_env(args.cam, not args.nocam)
    results = []
    for i in parse_states(args.states):
        env.reset()
        obs = env.set_init_state(inits[i])
        for _ in range(args.settle):
            obs, _, _, _ = env.step(np.zeros(7))
        frames = []
        on_step = None
        if args.video:
            frames.append(obs["agentview_image"][::-1].copy())
            on_step = lambda o: frames.append(o["agentview_image"][::-1].copy())
        t0 = time.time()
        if args.v:
            print(f"state {i}:")
        r = C.run_episode(env, max_steps=args.max_steps, on_step=on_step, verbose=args.v,
                          stop_on_success=not args.no_stop)
        r["state"] = i
        r["time"] = time.time() - t0
        results.append(r)
        msg = (f"state {i}: {'SUCCESS at step %d' % r['success_step'] if r['success'] else 'FAIL'}"
               f"  (stage {r['stage']}, steps {r['steps']}, knob {r['knob']:.2f}, "
               f"pan {np.round(r['pan'], 3)}, final_success {r['final_success']}, "
               f"released@{r['release_step']}, {r['time']:.1f}s)")
        print(msg, flush=True)
        if args.video:
            import imageio
            os.makedirs(args.video, exist_ok=True)
            tag = "success" if r["success"] else "fail"
            path = os.path.join(args.video, f"state{i:02d}_{tag}.mp4")
            imageio.mimsave(path, frames, fps=20)
            print("  saved", path)
    n = sum(r["success"] for r in results)
    print(f"TOTAL {n}/{len(results)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
                       for r in results], f, indent=1)


if __name__ == "__main__":
    main()
