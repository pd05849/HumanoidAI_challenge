#!/usr/bin/env python
"""(a) Build the "moka" dataset: every episode of LIBERO-10 task 2, "turn on the stove and put the
moka pot on it", taken out of the LeRobot LIBERO dataset that smolvla_libero was trained on.

Episodes are selected by their task string (meta/episodes "tasks" column), so the script works
with any LeRobot v3 LIBERO dataset (HuggingFaceVLA/libero by default; lerobot/libero or a
per-suite dataset with --repo-id). Only the data/video files that contain those episodes are
downloaded. The subset is written with LeRobot's own split_dataset (the same code as
`lerobot-edit-dataset --operation.type split`), so features, fps, encoder settings and per-episode
stats are copied unchanged.

usage (LeRobot env with Hub access):
  python train/make_moka_subset.py --out outputs/datasets/moka
  python train/make_moka_subset.py --list-tasks                 # show the task strings and counts
  python train/make_moka_subset.py --repo-id lerobot/libero --out outputs/datasets/moka

Writes OUT/ (a LeRobot dataset), OUT_subset.json next to it (selected source episode indices,
lengths) and OUT_preview.png (first frame of both cameras, to check the image orientation).
NB the OpenVLA-regenerated LIBERO data (which HuggingFaceVLA/libero is built from) dropped demos
that failed on replay, so expect fewer than 50 moka episodes (the script prints the count).
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TASK_MOKA, json_default, normalize_task, read_info  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="HuggingFaceVLA/libero", help="source LeRobot LIBERO dataset")
    ap.add_argument("--src-root", default=None,
                    help="local folder for the (partial) source download (default: $HF_LEROBOT_HOME/<repo-id>)")
    ap.add_argument("--revision", default=None, help="source dataset revision (default: LeRobot's codebase tag)")
    ap.add_argument("--task", default=TASK_MOKA, help="task string to keep (case/whitespace-insensitive)")
    ap.add_argument("--out", default="outputs/datasets/moka", help="output dataset folder (must not exist)")
    ap.add_argument("--overwrite", action="store_true", help="delete --out first")
    ap.add_argument("--max-episodes", type=int, default=None, help="keep only the first N matching episodes")
    ap.add_argument("--list-tasks", action="store_true", help="print every task string with its episode count, exit")
    ap.add_argument("--dry-run", action="store_true", help="select episodes and print them, do not download data")
    args = ap.parse_args()

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.utils.constants import HF_LEROBOT_HOME

    src_root = Path(args.src_root) if args.src_root else HF_LEROBOT_HOME / args.repo_id
    try:
        meta = LeRobotDatasetMetadata(args.repo_id, root=src_root, revision=args.revision)
    except Exception as ex:  # most often: a v2.x dataset
        raise SystemExit(f"cannot load the metadata of {args.repo_id}: {type(ex).__name__}: {ex}\n"
                         f"If it is a LeRobot v2.x dataset, convert it first:\n"
                         f"  python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id={args.repo_id}")
    info = read_info(meta.root)
    print(f"{args.repo_id}: codebase {info.get('codebase_version')}, fps {info['fps']}, robot_type "
          f"{info.get('robot_type')}, {info['total_episodes']} episodes, {info['total_frames']} frames, "
          f"{info.get('total_tasks')} tasks, features {[k for k in info['features']]}")

    ep_tasks = meta.episodes["tasks"]
    ep_index = list(meta.episodes["episode_index"])
    ep_len = list(meta.episodes["length"])
    if args.list_tasks:
        cnt = Counter(t for ts in ep_tasks for t in ts)
        for t, c in sorted(cnt.items(), key=lambda x: x[0]):
            print(f"  {c:4d}  {t}")
        return

    want = normalize_task(args.task)
    sel = [(int(i), int(n)) for i, ts, n in zip(ep_index, ep_tasks, ep_len)
           if any(normalize_task(t) == want for t in ts)]
    mixed = [i for i, ts in zip(ep_index, ep_tasks) if len(ts) > 1 and any(normalize_task(t) == want for t in ts)]
    if not sel:
        near = sorted({t for ts in ep_tasks for t in ts if "moka" in t.lower() or "stove" in t.lower()})
        raise SystemExit(f"no episode has the task {args.task!r}. Tasks mentioning moka/stove:\n  "
                         + "\n  ".join(near))
    if mixed:
        print(f"WARNING: episodes {mixed} carry more than one task; they are kept whole")
    if args.max_episodes:
        sel = sel[:args.max_episodes]
    ids = [i for i, _ in sel]
    lens = [n for _, n in sel]
    print(f"selected {len(ids)} episodes of {args.task!r}: {len(ids)} episodes, {sum(lens)} frames, "
          f"length min/median/max {min(lens)}/{sorted(lens)[len(lens) // 2]}/{max(lens)}")
    print(f"  source episode indices: {ids}")
    if len(ids) != 50:
        print(f"  NOTE: {len(ids)} != 50 (the OpenVLA no-noop LIBERO data dropped demos that failed on replay)")
    if args.dry_run:
        return

    from lerobot.datasets.dataset_tools import split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(args.out).resolve()
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} already exists (pass --overwrite)")
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Downloads only the parquet/mp4 files that hold these episodes.
    src = LeRobotDataset(args.repo_id, root=src_root, episodes=ids, revision=args.revision)
    # split_dataset writes <output_dir>/<split name>; videos shared with other episodes are re-encoded
    # with the source's own encoder settings, everything else is copied as is.
    staging = out.parent / f".{out.name}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    split_dataset(src, splits={"subset": ids}, output_dir=staging)
    shutil.move(str(staging / "subset"), str(out))
    shutil.rmtree(staging, ignore_errors=True)

    if not (out / "meta" / "stats.json").exists():
        print("no stats.json after the split (source without per-episode stats): recomputing")
        from lerobot.datasets.dataset_tools import recompute_stats
        recompute_stats(LeRobotDataset("local/moka", root=out), skip_image_video=True)

    ds = LeRobotDataset("local/moka", root=out)
    tasks = list(ds.meta.tasks.index)
    print(f"wrote {out}: {ds.num_episodes} episodes, {ds.num_frames} frames, fps {ds.fps}, tasks {tasks}")
    if ds.num_episodes != len(ids) or [normalize_task(t) for t in tasks] != [want]:
        raise SystemExit("unexpected subset content")

    report = dict(source_repo_id=args.repo_id, source_revision=src.revision, source_root=str(src.root),
                  task=args.task, source_episode_indices=ids, lengths=lens, n_episodes=len(ids),
                  n_frames=int(sum(lens)), fps=info["fps"], robot_type=info.get("robot_type"),
                  features={k: {kk: vv for kk, vv in v.items() if kk != "info"} for k, v in info["features"].items()})
    rep = out.parent / f"{out.name}_subset.json"
    with open(rep, "w") as f:
        json.dump(report, f, indent=1, default=json_default)
    print(f"report {rep}")
    try:
        from check_datasets import save_preview
        png = out.parent / f"{out.name}_preview.png"
        save_preview([out], png)
        print(f"preview {png}  (upright scene expected: same orientation as our converted frames)")
    except Exception as ex:  # preview is a convenience
        print(f"(preview skipped: {ex})")


if __name__ == "__main__":
    main()
