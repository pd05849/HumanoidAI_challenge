#!/usr/bin/env python
"""Check that LeRobot datasets can be merged and look alike, before training on them.

For every dataset: fps, robot_type, episode/frame counts, tasks (with episode counts), features,
episode lengths, and action/state statistics read from the parquet data (mean, std, fraction of
saturated actions |a| > 0.999, fraction of frames with the gripper commanded closed).
Every dataset is checked against the FIRST one with LeRobot's merge rule (same fps, robot_type
and features; video encoder tuning keys ignored). Exit code 1 if any is incompatible.
--png saves frame 0 and a middle frame of episode 0 of every dataset (both cameras) side by
side: the scenes must have the same orientation (the robot arm enters from the same side).

usage:
  python train/check_datasets.py outputs/datasets/moka outputs/datasets/teleop_pan \
      outputs/datasets/phone_pan --png outputs/datasets/check.png
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import compare_features, harmonize_info, read_info, schema_from_like  # noqa: E402


def load_frames_table(root: Path):
    import pandas as pd
    files = sorted(glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet")))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def numeric_summary(root: Path, info: dict) -> dict:
    df = load_frames_table(root)
    if df is None:
        return {}
    out = {}
    lens = df.groupby("episode_index").size().to_numpy()
    out["episode_length"] = dict(min=int(lens.min()), median=float(np.median(lens)), max=int(lens.max()))
    for key in ("action", "observation.state"):
        if key not in df:
            continue
        x = np.stack(df[key].to_numpy()).astype(np.float64)
        s = dict(mean=x.mean(0).round(4).tolist(), std=x.std(0).round(4).tolist())
        if key == "action":
            s["saturated_frac"] = (np.abs(x[:, :6]) > 0.999).mean(0).round(3).tolist()
            s["gripper_closed_frac"] = float((x[:, 6] > 0).mean().round(3))
        out[key] = s
    return out


def save_preview(roots, png, n_cols_frames=(0, 0.5)):
    """Frame 0 and a middle frame of episode 0 of every dataset, both cameras, one row each."""
    from PIL import Image, ImageDraw
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rows = []
    for root in roots:
        root = Path(root)
        ds = LeRobotDataset("local/preview", root=root, episodes=[0])
        cams = list(ds.meta.camera_keys)
        n = ds.num_frames
        tiles = []
        for frac in n_cols_frames:
            item = ds[min(n - 1, int(frac * n))]
            for cam in cams:
                x = item[cam]
                x = (x.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy() if x.dtype.is_floating_point \
                    else x.permute(1, 2, 0).numpy()
                tiles.append((f"{cam.split('.')[-1]} t={min(n - 1, int(frac * n))}", Image.fromarray(x)))
        rows.append((root.name, tiles))
    W = 256
    head = 18
    ncol = max(len(t) for _, t in rows)
    canvas = Image.new("RGB", (ncol * W, len(rows) * (W + head)), (252, 252, 251))
    draw = ImageDraw.Draw(canvas)
    for r, (name, tiles) in enumerate(rows):
        y = r * (W + head)
        for c, (label, im) in enumerate(tiles):
            canvas.paste(im.resize((W, W)), (c * W, y + head))
            draw.text((c * W + 4, y + 3), f"{name}: {label}", fill=(20, 20, 20))
    canvas.save(png)
    return png


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="dataset folders; the first one is the reference")
    ap.add_argument("--png", default=None, help="save a side-by-side preview image")
    ap.add_argument("--json", default=None, help="save the summary as JSON")
    ap.add_argument("--no-stats", action="store_true", help="skip reading the parquet data")
    ap.add_argument("--harmonize", action="store_true",
                    help="make the metadata of datasets 2..N match the first one (see above)")
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots]
    if args.harmonize:
        schema = schema_from_like(roots[0])
        for r in roots[1:]:
            print(f"harmonize {r}: changed {harmonize_info(r, schema) or 'nothing'}")
    infos = [read_info(r) for r in roots]
    ref = infos[0]
    ok, summary = True, []
    for root, info in zip(roots, infos):
        import pandas as pd
        tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
        eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta" / "episodes" / "*" / "*.parquet")))],
                        ignore_index=True)
        task_counts = {}
        for ts in eps["tasks"]:
            for t in ts:
                task_counts[t] = task_counts.get(t, 0) + 1
        s = dict(root=str(root), codebase_version=info.get("codebase_version"), fps=info["fps"],
                 robot_type=info.get("robot_type"), total_episodes=info["total_episodes"],
                 total_frames=info["total_frames"], tasks=task_counts,
                 features={k: dict(dtype=v["dtype"], shape=v["shape"], names=v.get("names"),
                                   **({"codec": (v.get("info") or {}).get("video.codec"),
                                       "pix_fmt": (v.get("info") or {}).get("video.pix_fmt")}
                                      if v["dtype"] == "video" else {}))
                           for k, v in info["features"].items()
                           if k not in ("timestamp", "frame_index", "episode_index", "index", "task_index")},
                 has_stats=(root / "meta" / "stats.json").exists())
        if len(tasks) != len(task_counts):
            s["tasks_parquet_rows"] = len(tasks)
        diffs = []
        if info is not ref:
            if info["fps"] != ref["fps"]:
                diffs.append(f"fps {info['fps']} vs {ref['fps']}")
            if info.get("robot_type") != ref.get("robot_type"):
                diffs.append(f"robot_type {info.get('robot_type')!r} vs {ref.get('robot_type')!r}")
            diffs += compare_features(ref["features"], info["features"])
        s["merge_with_first"] = "ok" if not diffs else diffs
        ok &= not diffs
        if not args.no_stats:
            s["numeric"] = numeric_summary(root, info)
        summary.append(s)

        print(f"\n== {root}  ({s['codebase_version']}, fps {s['fps']}, robot_type {s['robot_type']}, "
              f"{s['total_episodes']} episodes, {s['total_frames']} frames, stats.json {s['has_stats']})")
        for t, c in task_counts.items():
            print(f"   task: {c:3d} x {t!r}")
        for k, v in s["features"].items():
            print(f"   {k}: {v}")
        if "numeric" in s and s["numeric"]:
            nm = s["numeric"]
            print(f"   episode length min/median/max: {nm['episode_length']}")
            if "action" in nm:
                print(f"   action mean {nm['action']['mean']}\n   action std  {nm['action']['std']}\n"
                      f"   action |a|>0.999 frac (xyz rpy) {nm['action']['saturated_frac']}, "
                      f"gripper closed frac {nm['action']['gripper_closed_frac']}")
            if "observation.state" in nm:
                print(f"   state mean  {nm['observation.state']['mean']}\n   state std   {nm['observation.state']['std']}")
        if info is not ref:
            print("   merge with first: " + ("OK" if not diffs else "INCOMPATIBLE\n      " + "\n      ".join(diffs)))

    if args.png:
        try:
            save_preview(roots, args.png)
            print(f"\npreview {args.png}")
        except Exception as ex:
            print(f"\n(preview failed: {type(ex).__name__}: {ex})")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1, default=str)
    print("\nALL COMPATIBLE" if ok else "\nNOT COMPATIBLE (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
