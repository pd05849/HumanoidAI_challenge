#!/usr/bin/env python
"""Check one make_demos.py episode against LeRobot's own LIBERO eval path, and replay it open-loop.

usage (LeRobot venv with LIBERO, e.g. the 3.12 replica or the training machine):
  python sim/check_demo_conventions.py OUT/episodes/<take>_seed<S>.npz [--replay kept|full|both|none]
                                       [--png check.png]

1. Conventions. Builds lerobot.envs.libero.LiberoEnv (libero_90 task 21, obs_type
   pixels_agent_pos, the eval defaults), points its init-state table at the episode's saved
   init_state, resets it with the episode's layout seed (so the seeded hard reset places the
   stove fixture where the generator had it) and pushes the observation through LeRobot's eval
   preprocessing (envs.utils.preprocess_observation -> processor LiberoProcessorStep). The
   result must equal the episode's frame 0: same keys, image and image2 pixel-exact after
   x255, state to float32 rounding. Also reports the stove pose vs the saved one and saves a
   PNG (raw env image, what the policy sees, what the generator stored, difference).
2. Open-loop replay in fresh LiberoEnvs, from the same reset:
     kept : the stored (no-op-filtered, float32) actions only, as a dataset replay would send them
     full : every control step, with the dropped no-op steps re-inserted as zero motion + the
            previous gripper command (states are then compared with the stored ones)
   Success = LiberoEnv's info["is_success"] (env.check_success()), as in lerobot-eval.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

SUITE, TASK_ID = "libero_90", 21
STOVE = "flat_stove_1"


def make_lenv(init_state):
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    suite = benchmark.get_benchmark_dict()[SUITE]()
    lenv = LiberoEnv(task_suite=suite, task_id=TASK_ID, task_suite_name=SUITE, obs_type="pixels_agent_pos",
                     episode_index=0)
    lenv._init_states = np.asarray(init_state, np.float64)[None]   # our layout instead of the 50 eval ones
    lenv.init_state_id = 0
    return lenv


def stove_pose(lenv):
    e = lenv._env.env
    bid = e.obj_body_id[STOVE]
    M = e.sim.model._model
    return np.array(M.body_pos[bid]), np.array(M.body_quat[bid])


def set_stove(lenv, pos, quat, init_state):
    """Fallback if the seeded reset did not reproduce the stove: write the saved fixture pose into
    the model and redo LiberoEnv.reset's set_init_state + settle by hand."""
    from lerobot.envs.libero import get_libero_dummy_action
    e = lenv._env.env
    bid = e.obj_body_id[STOVE]
    M = e.sim.model._model
    M.body_pos[bid] = pos
    M.body_quat[bid] = quat
    raw = lenv._env.set_init_state(init_state)
    for _ in range(lenv.num_steps_wait):
        raw, _, _, _ = lenv._env.step(get_libero_dummy_action())
    for robot in lenv._env.robots:
        robot.controller.use_delta = True
    return lenv._format_raw_obs(raw)


def reset_like_generator(init_state, seed, saved_pos, saved_quat):
    lenv = make_lenv(init_state)
    t0 = time.time()
    fobs, _ = lenv.reset(seed=int(seed))
    sp, sq = stove_pose(lenv)
    d = dict(stove_pos_err_mm=float(np.abs(sp - saved_pos).max() * 1e3),
             stove_quat_err=float(np.abs(sq - saved_quat).max()), reset_s=time.time() - t0, stove_fixed=False)
    if d["stove_pos_err_mm"] > 1e-6 or d["stove_quat_err"] > 1e-9:
        fobs = set_stove(lenv, saved_pos, saved_quat, init_state)
        d["stove_fixed"] = True
    return lenv, fobs, d


def policy_view(fobs):
    """Exactly what lerobot-eval feeds the policy for one LiberoEnv observation (batch of 1)."""
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep
    b = {"pixels": {k: np.asarray(v)[None] for k, v in fobs["pixels"].items()},
         "robot_state": {g: {k: np.asarray(v)[None] for k, v in dd.items()} for g, dd in fobs["robot_state"].items()}}
    return LiberoProcessorStep().observation(preprocess_observation(b))


def to_u8(t):
    """(1,3,H,W) float in [0,1] -> (H,W,3) uint8 (exact inverse of preprocess_observation's /255)."""
    return (t[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def replay(init_state, seed, saved_pos, saved_quat, actions, states_ref=None, ref_steps=None):
    lenv, fobs, _ = reset_like_generator(init_state, seed, saved_pos, saved_quat)
    first, dev, t0 = None, [], time.time()
    pv = None
    try:
        for i, a in enumerate(actions):
            if states_ref is not None and ref_steps is not None and i in ref_steps:
                pv = policy_view(fobs)
                dev.append(float(np.abs(pv["observation.state"][0].numpy() - states_ref[ref_steps[i]]).max()))
            fobs, _, term, _, info = lenv.step(np.asarray(a, np.float32))
            if info["is_success"] and first is None:
                first = i
        final = bool(lenv._env.check_success())
    finally:
        lenv.close()
    return dict(first_success_index=first, final_success=final, n_actions=len(actions),
                max_state_dev=max(dev) if dev else None, state_dev_at_end=dev[-1] if dev else None,
                time_s=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("npz")
    ap.add_argument("--replay", default="both", choices=["kept", "full", "both", "none"])
    ap.add_argument("--png", default=None, help="orientation figure (default: next to the npz)")
    args = ap.parse_args()
    d = np.load(args.npz)
    with open(os.path.splitext(args.npz)[0] + ".json") as f:
        meta = json.load(f)
    seed, init = meta["seed"], d["init_state"]
    out = dict(episode=os.path.basename(args.npz), seed=seed, states_from=meta.get("states_from"))

    # ---- 1. conventions at frame 0
    lenv, fobs, sd = reset_like_generator(init, seed, d["stove_pos"], d["stove_quat"])
    out["reset"] = sd
    pv = policy_view(fobs)
    keys_policy = sorted(k for k in pv if k.startswith("observation."))
    out["policy_keys"] = keys_policy
    img_p, img2_p = to_u8(pv["observation.images.image"]), to_u8(pv["observation.images.image2"])
    st_p = pv["observation.state"][0].numpy()
    raw_img = np.asarray(fobs["pixels"]["image"])
    assert int(d["step"][0]) == 0, "frame 0 is not control step 0"
    cmp = dict(
        image_max_abs=int(np.abs(img_p.astype(int) - d["image"][0]).max()),
        image2_max_abs=int(np.abs(img2_p.astype(int) - d["image2"][0]).max()),
        image_vs_raw_unflipped_mean=float(np.abs(raw_img.astype(int) - d["image"][0]).mean()),
        image_vs_raw_flipH_only_mean=float(np.abs(raw_img[::-1].astype(int) - d["image"][0]).mean()),
        state_max_abs=float(np.abs(st_p - d["state"][0]).max()),
        state_bit_exact=bool(np.array_equal(st_p.view(np.uint32), d["state"][0].view(np.uint32))),
        state_dtype=str(pv["observation.state"].dtype), state_shape=list(pv["observation.state"].shape[1:]),
        image_shape=list(pv["observation.images.image"].shape[1:]))
    # same check on the raw robot_state for the parts that are copied (pos, gripper)
    cmp["eef_pos_max_abs"] = float(np.abs(np.asarray(fobs["robot_state"]["eef"]["pos"], np.float32) - d["state"][0][:3]).max())
    cmp["gripper_qpos_max_abs"] = float(np.abs(np.asarray(fobs["robot_state"]["gripper"]["qpos"], np.float32) - d["state"][0][6:]).max())
    out["frame0"] = cmp
    lenv.close()
    png = args.png or os.path.splitext(args.npz)[0] + "_conventions.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 5, figsize=(16, 3.6))
        diff = np.abs(img_p.astype(int) - d["image"][0]).sum(-1)
        for a, im, t in zip(ax, [raw_img, img_p, d["image"][0], d["image2"][0], diff],
                            ["LiberoEnv raw obs (agentview)", "policy input after\nLiberoProcessorStep",
                             "generator stored frame 0\n(observation.images.image)", "stored image2 (wrist)",
                             f"|policy - stored| (max {cmp['image_max_abs']})"]):
            a.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=max(1, int(im.max())) if im.ndim == 2 else None)
            a.set_title(t, fontsize=9)
            a.axis("off")
        fig.tight_layout()
        fig.savefig(png, dpi=90)
        plt.close(fig)
        out["png"] = png
    except Exception as ex:      # figure is optional
        out["png_error"] = repr(ex)

    # ---- 2. open-loop replays
    acts, steps = d["action"], d["step"].astype(int)
    fs_step, fs_frame = meta.get("first_success_step"), meta.get("first_success_frame")
    if args.replay in ("kept", "both"):
        r = replay(init, seed, d["stove_pos"], d["stove_quat"], acts,
                   states_ref=d["state"], ref_steps={i: i for i in range(len(acts))})
        r["expected_first_success_frame"] = fs_frame
        out["replay_kept"] = r
    if args.replay in ("full", "both"):
        n = int(meta.get("n_steps", steps[-1] + 1))
        full, kidx, prev = [], {}, None
        j = 0
        for s in range(n):
            if j < len(steps) and steps[j] == s:
                full.append(acts[j])
                kidx[s] = j
                prev = acts[j]
                j += 1
            else:                                    # dropped no-op: zero motion, gripper unchanged
                g = prev[-1] if prev is not None else -1.0
                full.append(np.array([0, 0, 0, 0, 0, 0, g], np.float32))
        r = replay(init, seed, d["stove_pos"], d["stove_quat"], full, states_ref=d["state"], ref_steps=kidx)
        r["expected_first_success_step"] = fs_step
        r["n_reinserted_noops"] = n - len(steps)
        out["replay_full"] = r
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
