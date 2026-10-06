"""Rewrite a LeRobot LIBERO-style dataset so it can be merged with a reference dataset.

    python train/conform_dataset.py SRC_DATASET --like outputs/datasets/moka --out outputs/datasets/phone_pan

Used for the phone-derived datasets, which are written in the cloud session without access to
the reference (the moka subset of the LeRobot LIBERO dataset): if their fps label, video codec or
feature names differ from the reference's, `lerobot-edit-dataset --operation.type merge` refuses.
This script decodes every episode (both cameras, state, action, task) and writes it again with
the reference's fps label, codec and feature names. Frames are not resampled: one frame is one
20 Hz control step in every LIBERO dataset, only the fps label differs. Re-encoding adds a small
compression error (about 1-2/255).
"""
import argparse
import glob
from pathlib import Path

import numpy as np

from common import (EpisodeWriter, KEY_ACTION, KEY_IMAGE, KEY_IMAGE2, KEY_STATE,  # noqa: F401
                    camera_keys, read_info, schema_from_like)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--like", required=True, help="reference LeRobot dataset (the moka subset)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default=None, help="default: local/<out folder name>")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import decode_video_frames

    src = Path(args.src)
    info = read_info(src)
    cam1, cam2 = camera_keys(info["features"])
    state_key = "observation.state"
    meta = LeRobotDatasetMetadata(f"local/{src.name}", root=src)
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(src / "data" / "*" / "*.parquet")))])
    tasks = pd.read_parquet(src / "meta" / "tasks.parquet")
    task_of = {int(r.task_index): t for t, r in tasks.iterrows()}

    schema = schema_from_like(args.like)
    writer = EpisodeWriter(args.out, args.repo_id or f"local/{Path(args.out).name}", schema,
                           overwrite=args.overwrite)
    fps = float(info["fps"])
    for ep in sorted(int(e) for e in df["episode_index"].unique()):
        e = df[df["episode_index"] == ep].sort_values("frame_index")
        ts = (e["frame_index"].to_numpy() / fps).tolist()
        epm = meta.episodes[ep]
        frames = {}
        for key in (cam1, cam2):
            path = src / meta.get_video_file_path(ep, key)
            start = float(epm[f"videos/{key}/from_timestamp"])
            fr = decode_video_frames(path, [start + t for t in ts], tolerance_s=0.5 / fps, return_uint8=True)
            frames[key] = fr.permute(0, 2, 3, 1).cpu().numpy()          # (T, H, W, 3) uint8
        state = np.stack(e[state_key].to_numpy()).astype(np.float32)
        action = np.stack(e["action"].to_numpy()).astype(np.float32)
        task = task_of[int(e["task_index"].iloc[0])]
        n = writer.add_episode(frames[cam1], frames[cam2], state, action, task)
        print(f"  episode {ep}: {n} frames", flush=True)
    changed = writer.close()
    print(f"wrote {args.out}: {writer.n_episodes} episodes, {writer.n_frames} frames, fps {schema.fps}"
          f"{'; harmonized ' + ', '.join(changed) if changed else ''}")


if __name__ == "__main__":
    main()
