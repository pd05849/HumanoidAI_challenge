#!/usr/bin/env python
"""Write make_demos.py episodes (episodes/<take>_seed<S>.npz + .json) as a LeRobot v3.0 dataset
(LeRobot 0.6.x, Python 3.12 venv), then load it back and check it.

usage:
  python sim/write_lerobot_dataset.py DEMOS_OUT [DEMOS_OUT2 | EPISODES_DIR | EP.npz ...] \
      --root OUT_DATASET_DIR [--repo-id local/libero90_frypan_phone] [--fps 20] \
      [--like SOURCE_DATASET/meta/info.json] [--trim-after-success N] [--overwrite]

Inputs: a make_demos.py --out folder (uses OUT/episodes/*.npz), an episodes folder, or .npz files.
Only episodes whose .json says kept=true and success=true are written. Episodes from the eval
init states (--states-from eval: those are the test layouts), episodes where the moka pot was
disturbed and episodes from make_demos.py versions before 2026-10-06b (robosuite init noise ->
controller posture target differing from LiberoEnv) are skipped unless --allow-eval-layouts /
--include-moka-disturbed / --allow-old-generator.

Features (the LeRobot LIBERO convention, = what LiberoEnv + LiberoProcessorStep feed a policy):
  observation.images.image   video (256,256,3)  agentview, already rotated 180 deg (npz "image")
  observation.images.image2  video (256,256,3)  wrist camera, already rotated 180 deg (npz "image2")
  observation.state          float32 (8,)       [eef_pos, axis-angle(eef_quat), gripper_qpos]
  action                     float32 (7,)       the env action (gripper -1 open, +1 close)
  task                       "turn on the stove and put the frying pan on it"
If the npz carries the raw robot state (eef_pos_raw, eef_quat_raw, gripper_qpos_raw; generator
versions from 2026-10-06 on), observation.state is recomputed here with LeRobot's own
LiberoProcessorStep._quat2axisangle, i.e. bit-identical to the eval pipeline; otherwise the
stored state (numpy float32, equal to it within 1 ulp) is used.

fps: the LIBERO control rate is 20 Hz (LiberoEnv control_freq). Merging with another dataset
(lerobot-edit-dataset --operation.type merge / dataset_tools.merge_datasets) requires the SAME
fps, robot_type and feature dicts (names, dtype, shape, video codec/pix_fmt/fps). Pass the source
dataset's meta/info.json with --like to copy its fps, robot_type, feature names/dtypes and video
codec (recommended: e.g. ~/.cache/huggingface/lerobot/HuggingFaceVLA/libero/meta/info.json); the
script then also runs LeRobot's own merge-compatibility check against it. fps only relabels the
timestamps (frame k -> k/fps); every recorded control step stays one frame.

--trim-after-success N keeps frames up to the first success + N (default -1 = keep all: the
release, retreat and idle steps after the pan first touches the stove). LeRobot eval stops an
episode at the first success; LIBERO's teleop demos end shortly after it.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np

TASK = "turn on the stove and put the frying pan on it"
CAM_KEYS = {"image": "observation.images.image", "image2": "observation.images.image2"}
STATE_KEY, ACTION_KEY = "observation.state", "action"
V = {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channels"]}
DEFAULT_FEATURES = {
    "observation.images.image": dict(V),
    "observation.images.image2": dict(V),
    STATE_KEY: {"dtype": "float32", "shape": (8,), "names": None},
    ACTION_KEY: {"dtype": "float32", "shape": (7,), "names": None},
}
AUTO_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
LIBERO90_MAX_STEPS = 400          # lerobot.envs.libero.TASK_SUITE_MAX_STEPS["libero_90"]


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_episodes(inputs):
    out = []
    for p in inputs:
        if os.path.isfile(p) and p.endswith(".npz"):
            out.append(os.path.abspath(p))
        elif os.path.isdir(os.path.join(p, "episodes")):
            out += sorted(glob.glob(os.path.join(p, "episodes", "*.npz")), key=natural_key)
        elif os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*.npz")), key=natural_key)
        else:
            print(f"WARNING: {p}: no episodes found")
    return [os.path.abspath(p) for p in out if not p.endswith(".tmp.npz")]


def lerobot_state(d):
    """observation.state from the raw robot state with LeRobot's own conversion, when stored."""
    if not all(k in d for k in ("eef_pos_raw", "eef_quat_raw", "gripper_qpos_raw")):
        return d["state"].astype(np.float32), "stored"
    import torch
    from lerobot.processor.env_processor import LiberoProcessorStep
    aa = LiberoProcessorStep()._quat2axisangle(torch.from_numpy(np.asarray(d["eef_quat_raw"], np.float64)))
    st = torch.cat((torch.from_numpy(np.asarray(d["eef_pos_raw"], np.float64)), aa,
                    torch.from_numpy(np.asarray(d["gripper_qpos_raw"], np.float64))), dim=-1).float()
    return st.numpy(), "recomputed (LiberoProcessorStep)"


def select(npz_paths, args):
    eps, seen, skipped = [], set(), []
    for p in npz_paths:
        jp = os.path.splitext(p)[0] + ".json"
        if not os.path.exists(jp):
            skipped.append((p, "no json"))
            continue
        with open(jp) as f:
            m = json.load(f)
        name = m.get("name") or os.path.splitext(os.path.basename(p))[0]
        why = None
        if not (m.get("kept") and m.get("success")):
            why = "not a kept success"
        elif m.get("task") != TASK:
            why = f"task {m.get('task')!r}"
        elif m.get("generator_version") is None and not args.allow_old_generator:
            why = ("made before the eval-identical-env fix of make_demos.py (2026-10-06b): its physics "
                   "differ from LiberoEnv, regenerate it (--allow-old-generator to include anyway)")
        elif m.get("states_from") == "eval" and not args.allow_eval_layouts:
            why = "layout is an eval init state (--allow-eval-layouts to include)"
        elif (m.get("moka") or {}).get("disturbed") and not args.include_moka_disturbed:
            why = (f"moka pot disturbed (moved {m['moka']['max_moved_mm']:.0f} mm, tilted "
                   f"{m['moka']['max_tilt_deg']:.0f} deg; --include-moka-disturbed to include)")
        elif name in seen:
            why = "duplicate episode name"
        if why:
            skipped.append((p, why))
            continue
        seen.add(name)
        eps.append((p, m))
    return eps, skipped


def features_from_like(info, key_map):
    """Our four features with the source dataset's names / dtype / video info; plus fps, robot_type,
    and the encoder config for its video codec."""
    src = info["features"]
    feats, missing = {}, []
    for ours, spec in DEFAULT_FEATURES.items():
        theirs = key_map.get(ours, ours)
        if theirs not in src:
            missing.append(theirs)
            continue
        s = src[theirs]
        if tuple(s["shape"]) != tuple(spec["shape"]):
            raise SystemExit(f"--like: {theirs} has shape {s['shape']}, our data is {spec['shape']}")
        if (s["dtype"] in ("video", "image")) != (spec["dtype"] == "video"):
            raise SystemExit(f"--like: {theirs} dtype {s['dtype']} vs ours {spec['dtype']}")
        if s["dtype"] not in ("video", "image") and s["dtype"] != spec["dtype"]:
            raise SystemExit(f"--like: {theirs} dtype {s['dtype']} vs ours {spec['dtype']}")
        feats[theirs] = {"dtype": s["dtype"], "shape": tuple(s["shape"]), "names": s.get("names")}
    if missing:
        raise SystemExit(f"--like: source lacks {missing}; its features are {sorted(src)}. "
                         f"Map ours to theirs with --key-map OURS=THEIRS.")
    extra = sorted(set(src) - set(feats) - set(AUTO_KEYS))
    if extra:
        print(f"WARNING: the source has features we do not write: {extra}. A merge will refuse "
              f"(feature sets must be identical).")
    vinfo = next((src[k].get("info") for k in feats if src[k]["dtype"] == "video" and src[k].get("info")), None)
    return feats, int(info["fps"]), info.get("robot_type"), vinfo


def build(args):
    t0 = time.time()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    paths = find_episodes(args.inputs)
    eps, skipped = select(paths, args)
    for p, why in skipped:
        print(f"skip {os.path.basename(p)}: {why}")
    if args.max_episodes:
        eps = eps[:args.max_episodes]
    if not eps:
        raise SystemExit("no episodes to write")
    key_map = dict(kv.split("=", 1) for kv in args.key_map)
    for k in key_map:
        if k not in DEFAULT_FEATURES:
            raise SystemExit(f"--key-map: {k} is not one of {list(DEFAULT_FEATURES)}")
    rgb_encoder = None
    if args.like:
        with open(args.like) as f:
            like = json.load(f)
        features, fps, robot_type, vinfo = features_from_like(like, key_map)
        if args.fps is not None and args.fps != fps:
            print(f"WARNING: --fps {args.fps} overrides the source's {fps}; a merge will refuse")
            fps = args.fps
        robot_type = args.robot_type or robot_type
        if vinfo:
            try:
                from lerobot.configs.video import RGBEncoderConfig
                rgb_encoder = RGBEncoderConfig.from_video_info(vinfo)
            except Exception as ex:
                print(f"WARNING: cannot reproduce the source's video encoder ({ex!r}); using LeRobot's default")
    else:
        features = {key_map.get(k, k): dict(v) for k, v in DEFAULT_FEATURES.items()}
        fps = args.fps if args.fps is not None else 20
        robot_type = args.robot_type or "panda"
    kimg, kimg2 = key_map.get(CAM_KEYS["image"], CAM_KEYS["image"]), key_map.get(CAM_KEYS["image2"], CAM_KEYS["image2"])
    kst, kact = key_map.get(STATE_KEY, STATE_KEY), key_map.get(ACTION_KEY, ACTION_KEY)

    root = os.path.abspath(args.root)
    if os.path.exists(root):
        if not args.overwrite:
            raise SystemExit(f"{root} exists (use --overwrite to replace it)")
        shutil.rmtree(root)
    os.makedirs(os.path.dirname(root), exist_ok=True)
    kw = dict(repo_id=args.repo_id, fps=fps, features=features, root=root, robot_type=robot_type,
              use_videos=any(f["dtype"] == "video" for f in features.values()))
    if rgb_encoder is not None:
        kw["rgb_encoder"] = rgb_encoder
    ds = LeRobotDataset.create(**kw)
    print(f"writing {len(eps)} episodes -> {root}  (repo_id {args.repo_id}, fps {fps}, robot_type {robot_type})",
          flush=True)

    sources = []
    for ei, (p, m) in enumerate(eps):
        te = time.time()
        d = np.load(p)
        T = int(len(d["action"]))
        for k, shp, dt in (("image", (T, 256, 256, 3), np.uint8), ("image2", (T, 256, 256, 3), np.uint8),
                           ("state", (T, 8), np.float32), ("action", (T, 7), np.float32)):
            if d[k].shape != shp or d[k].dtype != dt:
                raise SystemExit(f"{p}: {k} is {d[k].shape} {d[k].dtype}, expected {shp} {np.dtype(dt)}")
        state, state_src = lerobot_state(d)
        dev = float(np.abs(state - d["state"]).max())
        if dev > 1e-5:
            raise SystemExit(f"{p}: recomputed state differs from the stored one by {dev}")
        fsf = m.get("first_success_frame")
        n = T if args.trim_after_success < 0 or fsf is None else min(T, int(fsf) + 1 + args.trim_after_success)
        img, img2, act = d["image"], d["image2"], d["action"]
        for t in range(n):
            ds.add_frame({kimg: img[t], kimg2: img2[t], kst: state[t], kact: act[t], "task": TASK})
        ds.save_episode()
        src = dict(episode_index=ei, name=m.get("name"), take=m.get("take"), seed=m.get("seed"),
                   states_from=m.get("states_from"), npz=p, n_frames=n, n_frames_npz=T,
                   n_steps=m.get("n_steps"), first_success_frame=fsf, first_success_step=m.get("first_success_step"),
                   state_source=state_src, state_vs_stored_max_abs=dev, replay_config=m.get("replay_config"),
                   grasp_choice=(m.get("info") or {}).get("grasp_choice"), moka_disturbed=(m.get("moka") or {}).get("disturbed"),
                   eval_distance_cm=(m.get("eval_distance") or {}).get("min_joint_cm"))
        sources.append(src)
        print(f"  ep {ei:3d} {m.get('name', ''):22s} {n:4d} frames (npz {T}, first success at frame {fsf}) "
              f"state {state_src} | {time.time() - te:.1f}s", flush=True)
    ds.finalize()
    with open(os.path.join(root, "meta", "phone_demo_sources.json"), "w") as f:
        json.dump(dict(task=TASK, fps=fps, robot_type=robot_type, features=sorted(features), like=args.like,
                       trim_after_success=args.trim_after_success, episodes=sources, skipped=skipped),
                  f, indent=1, default=str)
    print(f"wrote {len(sources)} episodes, {sum(s['n_frames'] for s in sources)} frames in {time.time() - t0:.0f}s",
          flush=True)
    return root, sources, dict(img=kimg, img2=kimg2, state=kst, action=kact), features, fps, robot_type


def verify(root, repo_id, sources, keys, fps, robot_type, like=None, png=None):
    """Reload with LeRobotDataset and compare against the npz files."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    t0 = time.time()
    rd = LeRobotDataset(repo_id, root=root)
    res = dict(num_episodes=int(rd.num_episodes), num_frames=int(rd.num_frames), fps=rd.fps,
               robot_type=rd.meta.robot_type, tasks=list(rd.meta.tasks.index), problems=[])
    P = res["problems"]
    if rd.num_episodes != len(sources):
        P.append(f"num_episodes {rd.num_episodes} != {len(sources)}")
    if rd.num_frames != sum(s["n_frames"] for s in sources):
        P.append(f"num_frames {rd.num_frames} != {sum(s['n_frames'] for s in sources)}")
    if rd.fps != fps or rd.meta.robot_type != robot_type:
        P.append(f"fps/robot_type {rd.fps}/{rd.meta.robot_type} != {fps}/{robot_type}")
    if res["tasks"] != [TASK]:
        P.append(f"tasks {res['tasks']}")
    feats = rd.meta.features
    res["features"] = {k: dict(dtype=v["dtype"], shape=list(v["shape"])) for k, v in feats.items()}
    for k, shp, dt in ((keys["img"], [256, 256, 3], "video"), (keys["img2"], [256, 256, 3], "video"),
                       (keys["state"], [8], "float32"), (keys["action"], [7], "float32")):
        if k not in feats or list(feats[k]["shape"]) != shp or feats[k]["dtype"] not in (dt, "image"):
            P.append(f"feature {k}: {feats.get(k)}")
    lengths = [int(e["length"]) for e in rd.meta.episodes] if rd.meta.episodes is not None else []
    if lengths and lengths != [s["n_frames"] for s in sources]:
        P.append(f"episode lengths {lengths} != {[s['n_frames'] for s in sources]}")
    img_err, worst = [], 0.0
    start, shown = 0, None
    for s in sources:
        d = np.load(s["npz"])
        n = s["n_frames"]
        picks = sorted({0, n // 2, n - 1} | ({min(n - 1, s["first_success_frame"])} if s["first_success_frame"] is not None else set()))
        for t in picks:
            it = rd[start + t]
            if int(it["episode_index"]) != s["episode_index"] or int(it["frame_index"]) != t:
                P.append(f"ep {s['episode_index']} frame {t}: index mismatch {int(it['episode_index'])}/{int(it['frame_index'])}")
            if abs(float(it["timestamp"]) - t / fps) > 1e-4:
                P.append(f"ep {s['episode_index']} frame {t}: timestamp {float(it['timestamp'])} != {t / fps}")
            if it["task"] != TASK:
                P.append(f"ep {s['episode_index']}: task {it['task']!r}")
            st = it[keys["state"]].numpy()
            ref_state = d["state"][t] if s["state_source"] == "stored" else lerobot_state(d)[0][t]
            if not np.array_equal(st, ref_state):
                P.append(f"ep {s['episode_index']} frame {t}: state differs by {np.abs(st - ref_state).max()}")
            if not np.array_equal(it[keys["action"]].numpy(), d["action"][t]):
                P.append(f"ep {s['episode_index']} frame {t}: action differs")
            for k, nk in ((keys["img"], "image"), (keys["img2"], "image2")):
                x = it[k]
                if tuple(x.shape) != (3, 256, 256) or str(x.dtype) != "torch.float32":
                    P.append(f"{k}: decoded {tuple(x.shape)} {x.dtype}")
                dec = (x.permute(1, 2, 0).numpy() * 255.0)
                e = np.abs(dec - d[nk][t].astype(np.float32))
                img_err.append(float(e.mean()))
                worst = max(worst, float(e.mean()))
                # orientation: the decoded frame must match the stored one far better than its 180-deg rotation
                e_rot = float(np.abs(dec[::-1, ::-1] - d[nk][t].astype(np.float32)).mean())
                if e_rot < 3 * float(e.mean()) + 1.0:
                    P.append(f"ep {s['episode_index']} frame {t} {k}: orientation ambiguous ({e.mean():.1f} vs {e_rot:.1f})")
                if shown is None and nk == "image" and t == n // 2:
                    shown = (dec.round().clip(0, 255).astype(np.uint8), d[nk][t], s["name"], t)
        start += n
    res["image_mean_abs_err_255"] = dict(mean=float(np.mean(img_err)), worst_frame=worst)
    if worst > 8.0:
        P.append(f"video decode error up to {worst:.1f}/255")
    if like:
        try:
            from lerobot.datasets.feature_utils import features_equal_for_merge
            with open(like) as f:
                src = json.load(f)
            ok = features_equal_for_merge(src["features"], rd.meta.features)
            same = dict(features=bool(ok), fps=src.get("fps") == rd.fps, robot_type=src.get("robot_type") == rd.meta.robot_type)
            res["merge_compatible_with_like"] = same
            if not all(same.values()):
                P.append(f"not merge-compatible with {like}: {same}")
        except Exception as ex:
            P.append(f"merge check failed: {ex!r}")
    if png and shown is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            dec, ref, nm, t = shown
            fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
            for a, im, ti in zip(ax, [ref, dec, np.abs(dec.astype(int) - ref).sum(-1)],
                                 [f"npz {nm} frame {t}", "decoded from the LeRobot dataset", "|difference| (AV1)"]):
                a.imshow(im, cmap="magma" if im.ndim == 2 else None)
                a.set_title(ti, fontsize=9)
                a.axis("off")
            fig.tight_layout()
            fig.savefig(png, dpi=90)
            plt.close(fig)
            res["png"] = png
        except Exception as ex:
            res["png_error"] = repr(ex)
    fsf = [s["first_success_frame"] for s in sources if s["first_success_frame"] is not None]
    if fsf:
        res["first_success_frame"] = dict(min=min(fsf), max=max(fsf), mean=float(np.mean(fsf)))
        late = [s["name"] for s in sources if (s["first_success_frame"] or 0) > LIBERO90_MAX_STEPS - 40]
        if late:
            res["warning_late_success"] = (f"{len(late)} episodes first succeed after frame "
                                           f"{LIBERO90_MAX_STEPS - 40} (libero_90 eval horizon {LIBERO90_MAX_STEPS}): {late}")
    res["verify_s"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("inputs", nargs="+", help="make_demos.py --out folders, episodes folders or .npz files")
    ap.add_argument("--root", required=True, help="dataset folder to create")
    ap.add_argument("--repo-id", default="local/libero90_frypan_phone")
    ap.add_argument("--fps", type=int, default=None, help="dataset fps (default 20 = LIBERO control rate, or --like's)")
    ap.add_argument("--robot-type", default=None, help="default panda (or --like's)")
    ap.add_argument("--like", default=None, help="meta/info.json of the dataset this one will be merged with")
    ap.add_argument("--key-map", nargs="*", default=[], help="OURS=THEIRS feature renames, e.g. "
                    "observation.images.image2=observation.images.wrist_image")
    ap.add_argument("--trim-after-success", type=int, default=-1,
                    help="keep frames up to first success + N (-1 = keep all)")
    ap.add_argument("--include-moka-disturbed", action="store_true")
    ap.add_argument("--allow-eval-layouts", action="store_true")
    ap.add_argument("--allow-old-generator", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    root, sources, keys, features, fps, robot_type = build(args)
    if args.no_verify:
        return
    res = verify(root, args.repo_id, sources, keys, fps, robot_type, like=args.like,
                 png=os.path.join(os.path.dirname(root), os.path.basename(root) + "_decoded_check.png"))
    print(json.dumps(res, indent=1, default=str))
    print("VERIFY OK" if not res["problems"] else f"VERIFY FAILED: {len(res['problems'])} problem(s)")
    sys.exit(0 if not res["problems"] else 1)


if __name__ == "__main__":
    main()
