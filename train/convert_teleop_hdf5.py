#!/usr/bin/env python
"""(b) Convert the 50 human teleop demos of LIBERO-90 task 21, "turn on the stove and put the
frying pan on it", into a LeRobot dataset in the same layout as the moka subset.

The LIBERO HDF5 stores 128x128 images, so the demos are re-rendered at 256x256 in LIBERO:

  --mode actions  OpenVLA's recipe (regenerate_libero_dataset.py, which is how the LIBERO data that
                  smolvla_libero was trained on was made): set the demo's first MuJoCo state, 10
                  settle steps [0,0,0,0,0,0,-1] (as LeRobot's LiberoEnv.reset), then re-send the
                  human's actions; no-op actions are skipped (neither executed nor recorded); the
                  demo is kept only if LIBERO's check_success() holds at the end.
  --mode states   re-render every recorded MuJoCo state (exactly the human's motion), paired with
                  the recorded action; no-op frames dropped. Always reproduces the demo.
  --mode auto     (default) actions, and states for the demos whose action replay fails, so all
                  50 human demos are kept; the report says which mode each one used.

The stove is a LIBERO fixture: its position is drawn at every env reset (about +-1 cm) and is not
part of the MuJoCo state vector. Each demo's own stove pose is read from the demo's `model_file`
XML attribute and written into the model before replay (--no-fixture-fix to disable); without
it, the re-rendered stove is not where the human's robot turned the knob.

Per frame (what lerobot-eval feeds the policy, see train/common.py):
  image/image2 = agentview / eye-in-hand rotated 180 deg, state = [eef_pos, axis-angle, gripper_qpos],
  action = the recorded 7-D action. Frame t = observation before action t.

usage (LeRobot env with lerobot[libero]; MUJOCO_GL=egl on a GPU machine):
  python train/convert_teleop_hdf5.py --like outputs/datasets/moka --out outputs/datasets/teleop_pan
  python train/convert_teleop_hdf5.py --hdf5 path/to/KITCHEN_SCENE3_..._demo.hdf5 --demos 0-4 ...
Writes OUT/ (LeRobot dataset), OUT_report.json and OUT_preview.png next to it.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (TASK_PAN, EpisodeWriter, images, is_noop, json_default, natural_key,  # noqa: E402
                    normalize_task, schema_from_like, state8)

HF_REPO = "yifengzhu-hf/LIBERO-datasets"
HF_FILE = "libero_90/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it_demo.hdf5"
SUITE, TASK_ID, CAM = "libero_90", 21, 256
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]   # lerobot get_libero_dummy_action()
EVAL_MAX_STEPS = 400                                   # lerobot TASK_SUITE_MAX_STEPS["libero_90"]


# ============================================================================ LIBERO
def libero_task():
    from libero.libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(TASK_ID)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return bddl, task.language, np.asarray(suite.get_task_init_states(TASK_ID))


def make_env(bddl):
    from libero.libero.envs import OffScreenRenderEnv
    # as lerobot.envs.libero.LiberoEnv builds it (no robosuite initialization noise, 20 Hz)
    return OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=CAM, camera_widths=CAM, control_freq=20)


def _floats(s):
    return np.array([float(x) for x in s.split()], dtype=np.float64)


def body_pose_in_xml(xml: str, body: str):
    """pos (3,) and quat wxyz (4,) or None of <body name="body" ...> in an MJCF string."""
    m = re.search(r'<body\b[^>]*\bname="%s"[^>]*>' % re.escape(body), xml)
    if m is None:
        return None
    tag = m.group(0)
    pos = re.search(r'\bpos="([^"]+)"', tag)
    quat = re.search(r'\bquat="([^"]+)"', tag)
    return (_floats(pos.group(1)) if pos else np.zeros(3),
            _floats(quat.group(1)) if quat else None)


def apply_fixture_poses(env, demo_xml: str) -> dict:
    """Move every fixture (stove, ...) to where it was in the demo's model XML."""
    e = env.env
    sim = e.sim
    M = getattr(sim.model, "_model", sim.model)
    out = {}
    for name in getattr(e, "fixtures_dict", {}):
        bid = e.obj_body_id.get(name)
        if bid is None:
            continue
        bname = sim.model.body_id2name(bid)
        found = body_pose_in_xml(demo_xml, bname)
        if found is None:
            out[name] = dict(body=bname, status="not in demo xml")
            continue
        pos, quat = found
        before = np.array(M.body_pos[bid])
        M.body_pos[bid] = pos
        if quat is not None:
            M.body_quat[bid] = quat
        out[name] = dict(body=bname, demo_pos=pos.round(5).tolist(), reset_pos=before.round(5).tolist(),
                         moved_mm=float(np.linalg.norm(pos - before) * 1e3))
    sim.forward()
    return out


# ============================================================================ one demo
def replay_actions(env, states, actions, settle, noops):
    obs = env.set_init_state(states[0])
    for _ in range(settle):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    rec = dict(image=[], image2=[], state=[], action=[])
    prev, n_noop, first = None, 0, None
    for a in actions:
        if is_noop(a, prev):
            n_noop += 1
            if noops == "drop":                 # executed, not recorded
                obs, _, _, _ = env.step(a.tolist())
            continue                            # "skip": neither executed nor recorded (OpenVLA)
        im, im2 = images(obs)
        rec["image"].append(im)
        rec["image2"].append(im2)
        rec["state"].append(state8(obs))
        rec["action"].append(np.asarray(a, np.float32))
        prev = a
        obs, _, _, _ = env.step(a.tolist())
        if first is None and env.check_success():
            first = len(rec["action"]) - 1
    return rec, bool(env.check_success()), first, n_noop


def playback_states(env, states, actions):
    rec = dict(image=[], image2=[], state=[], action=[])
    prev, n_noop = None, 0
    for s, a in zip(states, actions):
        if is_noop(a, prev):
            n_noop += 1
            continue
        obs = env.regenerate_obs_from_state(s)
        im, im2 = images(obs)
        rec["image"].append(im)
        rec["image2"].append(im2)
        rec["state"].append(state8(obs))
        rec["action"].append(np.asarray(a, np.float32))
        prev = a
    env.regenerate_obs_from_state(states[-1])
    return rec, bool(env.check_success()), None, n_noop


def parse_demos(spec, names):
    if spec is None:
        return names
    want = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            want.update(range(int(a), int(b) + 1))
        elif part.strip():
            want.add(int(part))
    return [n for n in names if int(n.split("_")[-1]) in want]


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5", default=None, help=f"local demo file (default: download {HF_FILE} from {HF_REPO})")
    ap.add_argument("--like", default=None, help="reference LeRobot dataset to match (the moka subset)")
    ap.add_argument("--out", default="outputs/datasets/teleop_pan")
    ap.add_argument("--repo-id", default="local/teleop_pan")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--mode", choices=["auto", "actions", "states"], default="auto")
    ap.add_argument("--noops", choices=["skip", "drop"], default="skip",
                    help="actions mode: skip = no-ops neither executed nor recorded (OpenVLA); "
                         "drop = executed but not recorded (sim/make_demos.py)")
    ap.add_argument("--settle", type=int, default=10, help="dummy steps before the replay (LeRobot eval: 10)")
    ap.add_argument("--demos", default=None, help="e.g. 0-49 or 0,3,7 (default: all)")
    ap.add_argument("--seed", type=int, default=0, help="env seed for the reset before each demo")
    ap.add_argument("--no-fixture-fix", action="store_true", help="keep the stove where the reset put it")
    ap.add_argument("--reuse-env", action="store_true", help="one env for all demos (default: fresh env per demo)")
    ap.add_argument("--fps", type=int, default=None, help="dataset fps label (default: the reference's)")
    ap.add_argument("--vcodec", default=None, help="video codec (default: the reference's)")
    ap.add_argument("--video", action="store_true", help="also save OUT_episode0.mp4 (agentview)")
    args = ap.parse_args()

    import h5py
    path = args.hdf5
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
    print(f"demo file: {path}")

    bddl, language, eval_init = libero_task()
    if normalize_task(language) != normalize_task(TASK_PAN):
        raise SystemExit(f"{SUITE} task {TASK_ID} is {language!r}, expected {TASK_PAN!r}")
    schema = schema_from_like(args.like, fps=args.fps, vcodec=args.vcodec)
    out = Path(args.out).resolve()
    print(f"schema: fps {schema.fps}, robot_type {schema.robot_type}, keys {schema.key_map}"
          f"{'' if args.like else ' (no --like: defaults; the merge may refuse)'}")

    h = h5py.File(path, "r")
    data = h["data"]
    pinfo = data.attrs.get("problem_info")
    if pinfo:
        lang = json.loads(pinfo).get("language_instruction", "")
        lang = " ".join(lang) if isinstance(lang, list) else lang
        if lang and normalize_task(lang) != normalize_task(TASK_PAN):
            print(f"WARNING: the HDF5 says {lang!r}")
    names = parse_demos(args.demos, sorted(data.keys(), key=natural_key))
    print(f"{len(names)} demos; mode {args.mode}; task {language!r}")

    writer = EpisodeWriter(out, args.repo_id, schema, overwrite=args.overwrite)
    report, env, t_all = [], None, time.time()
    for i, name in enumerate(names):
        t0 = time.time()
        g = data[name]
        states, actions = g["states"][()], g["actions"][()]
        obs_keys = {k: tuple(g["obs"][k].shape) for k in g.get("obs", {}).keys()} if "obs" in g else {}
        if env is None or not args.reuse_env:
            if env is not None:
                env.close()
            env = make_env(bddl)
        env.seed(args.seed)
        env.reset()
        dim = env.env.sim.get_state().flatten().size
        if states.shape[1] != dim:
            raise SystemExit(f"{name}: state dim {states.shape[1]} != env state dim {dim} "
                             f"(LIBERO / robosuite version mismatch?)")
        fixtures = {}
        xml = g.attrs.get("model_file")
        if isinstance(xml, bytes):
            xml = xml.decode()
        if not args.no_fixture_fix:
            fixtures = apply_fixture_poses(env, xml) if xml else {"_": "no model_file attribute"}
        d_eval = np.linalg.norm(eval_init[:, 1:] - states[0][1:], axis=1)
        rec = dict(demo=name, n_steps=int(len(actions)), hdf5_obs_shapes=obs_keys, fixtures=fixtures,
                   nearest_eval_init_state=int(np.argmin(d_eval)), nearest_eval_init_dist=float(d_eval.min()))
        r, ok, first, n_noop = None, False, None, 0
        if args.mode in ("auto", "actions"):
            r, ok, first, n_noop = replay_actions(env, states, actions, args.settle, args.noops)
            rec.update(actions_success=ok, actions_first_success_frame=first)
            used = "actions"
        if args.mode == "states" or (args.mode == "auto" and not ok):
            if args.mode == "auto":          # fresh model state for the playback
                env.seed(args.seed)
                env.reset()
                if not args.no_fixture_fix and xml:
                    apply_fixture_poses(env, xml)
            r, ok_s, first, n_noop = playback_states(env, states, actions)
            rec.update(states_success_at_last_state=ok_s)
            used, ok = "states", True        # a recorded human demo: kept
        n = len(r["action"])
        rec.update(mode=used, kept=bool(ok), n_frames=n, n_noops=n_noop)
        if ok:
            writer.add_episode(r["image"], r["image2"], r["state"], r["action"], TASK_PAN)
            if args.video and writer.n_episodes == 1:
                import imageio
                vp = out.parent / f"{out.name}_episode0.mp4"
                imageio.mimsave(vp, r["image"], fps=20)
                print(f"  video {vp}")
        rec["time_s"] = time.time() - t0
        report.append(rec)
        fx = fixtures.get("flat_stove_1", {}) if isinstance(fixtures, dict) else {}
        print(f"  {name}: {used:7s} kept={ok!s:5s} frames {n:3d} (of {len(actions)} steps, {n_noop} no-ops)"
              f"{'' if first is None else f' success@{first}'} | stove moved {fx.get('moved_mm', float('nan')):.1f} mm"
              f" | nearest eval init {rec['nearest_eval_init_state']} ({rec['nearest_eval_init_dist']:.3f})"
              f" | {rec['time_s']:.0f}s", flush=True)
    if env is not None:
        env.close()
    h.close()
    changed = writer.close()

    kept = [r for r in report if r["kept"]]
    summary = dict(source=path, task=TASK_PAN, mode=args.mode, noops=args.noops, settle=args.settle,
                   n_demos=len(report), n_kept=len(kept),
                   n_actions_mode=sum(r["mode"] == "actions" for r in kept),
                   n_states_mode=sum(r["mode"] == "states" for r in kept),
                   n_frames=int(sum(r["n_frames"] for r in kept)),
                   over_eval_horizon=[r["demo"] for r in kept if r["n_frames"] > EVAL_MAX_STEPS],
                   fps=schema.fps, like=args.like, harmonized_info_keys=changed,
                   wall_s=time.time() - t_all, demos=report)
    rep = out.parent / f"{out.name}_report.json"
    with open(rep, "w") as f:
        json.dump(summary, f, indent=1, default=json_default)
    print(f"kept {len(kept)}/{len(report)} demos ({summary['n_actions_mode']} action replay, "
          f"{summary['n_states_mode']} state playback), {summary['n_frames']} frames -> {out}")
    print(f"report {rep}")
    if summary["over_eval_horizon"]:
        print(f"NOTE: longer than the {EVAL_MAX_STEPS}-step eval horizon: {summary['over_eval_horizon']}")
    close_eval = [r["demo"] for r in report if r["nearest_eval_init_dist"] < 1e-3]
    if close_eval:
        print(f"NOTE: {len(close_eval)} demos start from (almost) an eval init state: {close_eval[:10]}...")
    try:
        from check_datasets import save_preview
        png = out.parent / f"{out.name}_preview.png"
        save_preview([out], png)
        print(f"preview {png}")
    except Exception as ex:
        print(f"(preview skipped: {ex})")


if __name__ == "__main__":
    main()
