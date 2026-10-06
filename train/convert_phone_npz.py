#!/usr/bin/env python
"""(c) Write the phone-derived robot demos (sim/make_demos.py output) into a LeRobot dataset in
the same layout as the moka subset.

Input: one or more make_demos.py output folders (OUT/episodes/<take>_seed<S>.npz + .json) or
individual .npz files. Per episode, sim/make_demos.py already stores what lerobot-eval feeds the
policy (see its docstring): image/image2 (T,256,256,3) uint8 rotated 180 deg, state (T,8) float32
[eef_pos, axis-angle, gripper_qpos], action (T,7) float32, no-op frames dropped, one frame per
20 Hz control step; the .json has kept/success/task/first_success_frame. Only kept, successful
episodes are used.

Trimming: make_demos.py records idle steps after the pan is released. The LIBERO teleop demos end
when the task is done (robosuite's demo collection stops after ~10 successful steps), and an eval
episode ends at the first success, so frames more than --keep-after-success frames after the first
success are dropped (-1 keeps everything).

Selection: --max-episodes N keeps N episodes, taken round-robin over takes (Demo1 layout 0, Demo2
layout 0, ..., Demo1 layout 1, ...) so every take is represented; use it to match the number of
teleop episodes in run T.

usage (any env with lerobot):
  python train/convert_phone_npz.py /path/to/make_demos_out --like outputs/datasets/moka \
      --out outputs/datasets/phone_pan [--max-episodes 50]
Writes OUT/ (LeRobot dataset), OUT_report.json and OUT_preview.png next to it.
"""
import argparse
import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (TASK_PAN, EpisodeWriter, json_default, natural_key, normalize_task,  # noqa: E402
                    schema_from_like)

EVAL_MAX_STEPS = 400   # lerobot TASK_SUITE_MAX_STEPS["libero_90"]


def find_episodes(inputs):
    out = []
    for p in inputs:
        p = Path(p)
        if p.is_file() and p.suffix == ".npz":
            out.append(p)
        elif (p / "episodes").is_dir():
            out += [Path(x) for x in glob.glob(str(p / "episodes" / "*.npz"))]
        elif p.is_dir():
            out += [Path(x) for x in glob.glob(str(p / "*.npz"))]
        else:
            print(f"WARNING: {p}: not an .npz or a make_demos.py output folder")
    out = [x for x in out if not x.name.endswith(".tmp.npz")]
    return sorted(set(out), key=lambda x: natural_key(x.name))


def load_meta(npz: Path) -> dict:
    j = npz.with_suffix(".json")
    if not j.is_file():
        return {"name": npz.stem, "_no_json": True}
    with open(j) as f:
        return json.load(f)


def round_robin(eps, n):
    """eps: list of (meta, path) sorted by take then layout; pick n spreading over takes."""
    by_take = defaultdict(list)
    for e in eps:
        by_take[e[0].get("take", e[0]["name"])].append(e)
    takes = sorted(by_take, key=natural_key)
    out, k = [], 0
    while len(out) < n and any(by_take[t] for t in takes):
        t = takes[k % len(takes)]
        if by_take[t]:
            out.append(by_take[t].pop(0))
        k += 1
    return out


def validate(d, meta, path):
    T = len(d["action"])
    problems = []
    for k, shp, dt in (("image", (T, 256, 256, 3), np.uint8), ("image2", (T, 256, 256, 3), np.uint8),
                       ("state", (T, 8), np.float32), ("action", (T, 7), np.float32)):
        if k not in d.files:
            problems.append(f"missing '{k}'")
        elif d[k].shape != shp or d[k].dtype != dt:
            problems.append(f"{k} {d[k].shape} {d[k].dtype}, expected {shp} {np.dtype(dt)}")
    if "action" in d.files and np.abs(d["action"]).max() > 1.0 + 1e-6:
        problems.append("action outside [-1, 1]")
    if not np.isfinite(d["state"]).all() or not np.isfinite(d["action"]).all():
        problems.append("non-finite state/action")
    if meta.get("task") and normalize_task(meta["task"]) != normalize_task(TASK_PAN):
        problems.append(f"task {meta['task']!r}")
    if meta.get("fps") not in (None, 20):
        problems.append(f"recorded at {meta['fps']} Hz (expected one frame per 20 Hz control step)")
    if problems:
        raise SystemExit(f"{path}: " + "; ".join(problems))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="make_demos.py output folders and/or .npz files")
    ap.add_argument("--like", default=None, help="reference LeRobot dataset to match (the moka subset)")
    ap.add_argument("--out", default="outputs/datasets/phone_pan")
    ap.add_argument("--repo-id", default="local/phone_pan")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None, help="keep N episodes, round-robin over takes")
    ap.add_argument("--takes", default=None, help="comma-separated take names to use (default: all)")
    ap.add_argument("--keep-after-success", type=int, default=10,
                    help="frames kept after the first success frame (-1: keep all)")
    ap.add_argument("--allow-moka-disturbed", action="store_true",
                    help="also use episodes where the moka pot was knocked (moka.disturbed in the json)")
    ap.add_argument("--fps", type=int, default=None, help="dataset fps label (default: the reference's)")
    ap.add_argument("--vcodec", default=None, help="video codec (default: the reference's)")
    ap.add_argument("--dry-run", action="store_true", help="list the selected episodes, write nothing")
    args = ap.parse_args()

    files = find_episodes(args.inputs)
    eps, skipped = [], defaultdict(list)
    takes_wanted = set(args.takes.split(",")) if args.takes else None
    for p in files:
        m = load_meta(p)
        if m.get("_no_json"):
            skipped["no json"].append(p.name)
            continue
        if not (m.get("kept") and m.get("success")):
            skipped["not kept/successful"].append(p.name)
            continue
        if takes_wanted and m.get("take") not in takes_wanted:
            skipped["take not selected"].append(p.name)
            continue
        if (m.get("moka") or {}).get("disturbed") and not args.allow_moka_disturbed:
            skipped["moka pot disturbed"].append(p.name)
            continue
        eps.append((m, p))
    eps.sort(key=lambda e: (natural_key(e[0].get("take", "")), e[0].get("layout_index", 0), e[0].get("seed", 0)))
    if args.max_episodes is not None:
        eps = round_robin(eps, args.max_episodes)
    for why, names in skipped.items():
        print(f"skipped {len(names)} ({why}): {names[:8]}{' ...' if len(names) > 8 else ''}")
    per_take = defaultdict(int)
    for m, _ in eps:
        per_take[m.get("take", "?")] += 1
    print(f"{len(eps)} episodes from {len(per_take)} takes: {dict(per_take)}")
    if not eps:
        raise SystemExit("nothing to write")
    if args.dry_run:
        for m, p in eps:
            print(f"  {p.name}: {m.get('n_frames')} frames, first success frame {m.get('first_success_frame')}")
        return

    schema = schema_from_like(args.like, fps=args.fps, vcodec=args.vcodec)
    print(f"schema: fps {schema.fps}, robot_type {schema.robot_type}, keys {schema.key_map}"
          f"{'' if args.like else ' (no --like: defaults; the merge may refuse)'}")
    out = Path(args.out).resolve()
    writer = EpisodeWriter(out, args.repo_id, schema, overwrite=args.overwrite)
    report, t0 = [], time.time()
    for m, p in eps:
        with np.load(p) as d:
            validate(d, m, p)
            T = len(d["action"])
            fs = m.get("first_success_frame")
            end = T if (fs is None or args.keep_after_success < 0) else min(T, fs + 1 + args.keep_after_success)
            sl = slice(0, end)
            n = writer.add_episode(d["image"][sl], d["image2"][sl], d["state"][sl], d["action"][sl], TASK_PAN)
            steps = d["step"][sl] if "step" in d.files else np.arange(n)
        fss = m.get("first_success_step")
        rec = dict(name=m["name"], take=m.get("take"), seed=m.get("seed"), layout_index=m.get("layout_index"),
                   frames_in=T, frames_written=n, first_success_frame=fs, first_success_step=fss,
                   last_step=int(steps[-1]) if n else None,
                   eval_distance_cm=(m.get("eval_distance") or {}).get("min_joint_cm"),
                   source=str(p))
        report.append(rec)
        warn = ""
        if fss is not None and fss >= EVAL_MAX_STEPS:
            warn = f"  WARNING: first success at step {fss} >= eval horizon {EVAL_MAX_STEPS}"
        print(f"  {m['name']:24s} {n:4d}/{T} frames, first success frame {fs}{warn}", flush=True)
    changed = writer.close()
    lens = [r["frames_written"] for r in report]
    summary = dict(task=TASK_PAN, n_episodes=len(report), n_frames=int(sum(lens)),
                   length_min=min(lens), length_median=float(np.median(lens)), length_max=max(lens),
                   per_take=dict(per_take), keep_after_success=args.keep_after_success, fps=schema.fps,
                   like=args.like, harmonized_info_keys=changed, inputs=args.inputs, wall_s=time.time() - t0,
                   episodes=report)
    rep = out.parent / f"{out.name}_report.json"
    with open(rep, "w") as f:
        json.dump(summary, f, indent=1, default=json_default)
    print(f"wrote {len(report)} episodes, {summary['n_frames']} frames -> {out}\nreport {rep}")
    try:
        from check_datasets import save_preview
        png = out.parent / f"{out.name}_preview.png"
        save_preview([out], png)
        print(f"preview {png}")
    except Exception as ex:
        print(f"(preview skipped: {ex})")


if __name__ == "__main__":
    main()
