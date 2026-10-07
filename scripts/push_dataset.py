"""Publish a phone-derived dataset on the Hugging Face Hub, in LeRobot format.

    hf auth login                                   # once, with a token that can write
    python scripts/push_dataset.py datasets/phone_pan_all --repo-id <hf_user>/libero90_frying_pan_phone
    python scripts/push_dataset.py datasets/phone_pan_P   --repo-id <hf_user>/libero90_frying_pan_phone_50

(datasets/... = the folders written by train/convert_phone_npz.py, README "Reproduce" (c), step 4.)

Uses LeRobot's own push_to_hub, which uploads the dataset, writes its dataset card (README.md
on the Hub) and tags the upload with LeRobot's codebase version so that LeRobotDataset("<repo-id>")
can load it. --dry-run writes the card locally instead (README_preview.md next to the dataset) and
uploads nothing. Public by default; --private keeps it private.
"""
import argparse
import json
from pathlib import Path

GITHUB = "https://github.com/pd05849/HumanoidAI_challenge"

DESCRIPTION = """\
Robot demonstrations of LIBERO-90 task 21, **"turn on the stove and put the frying pan on it"**,
made from phone videos of a person moving a real frying pan in a real kitchen.

How each demo was made ([code and write-up]({url})):
1. A printed marker inside a real pan and a printed board on the stove were filmed with an iPad
   (18 takes, {n_takes} of which gave demos). The pan's 3-D path was tracked in every frame.
2. Each path was fitted onto LIBERO's kitchen (rotated, scaled, shifted onto the simulated pan
   and burner; played 1.5x faster; raised locally to clear the moka pot).
3. A scripted Franka Panda in LIBERO turned the stove knob, grasped the pan's handle and carried
   the pan along the path, with physics; only episodes that passed LIBERO's own success check
   and quality gates (path followed within 2 cm RMS, moka pot undisturbed) were kept.
4. Recorded in the conventions of `HuggingFaceVLA/libero` (the data `smolvla_libero` was trained
   on): `observation.images.image` (front) and `observation.images.image2` (wrist), 256x256,
   rotated 180 degrees as LeRobot's LIBERO processor does; `observation.state` 8-D (end-effector
   position, axis-angle, gripper joints); `action` 7-D; fps label 10, one frame per 20 Hz
   control step. Frames are stored as AV1 video (`HuggingFaceVLA/libero` stores images in the
   parquet files; LeRobot reads both).

{n_episodes} episodes, {n_frames} frames. The knob turn and the grasp are scripted; the carry
(where the pan goes, how high, how it turns, how fast) comes from the phone videos. None of the
episodes starts from LIBERO's 50 fixed evaluation layouts (nearest: 0.3-2.5 cm).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="local LeRobot dataset folder")
    ap.add_argument("--repo-id", required=True, help="<hf_user>/<dataset_name>")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--license", default="apache-2.0")
    ap.add_argument("--dry-run", action="store_true", help="write the dataset card locally, upload nothing")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import create_lerobot_dataset_card

    root = Path(args.root).resolve()
    if not (root / "meta" / "info.json").exists():
        found = sorted(p.name for p in root.iterdir()) if root.is_dir() else None
        raise SystemExit(f"{root} is not a complete LeRobot dataset (no meta/info.json). "
                         + (f"It contains: {found}. " if found is not None else "The folder does not exist. ")
                         + "Write it first with train/convert_phone_npz.py (README 'Reproduce' (c), step 4).")
    info = json.load(open(root / "meta" / "info.json"))
    report = root.parent / f"{root.name}_report.json"
    n_takes = len(json.load(open(report)).get("per_take", {})) if report.exists() else "17"
    text = DESCRIPTION.format(url=GITHUB, n_takes=n_takes, n_episodes=info["total_episodes"],
                              n_frames=info["total_frames"])
    tags = ["LIBERO", "libero_90", "frying-pan", "real-to-sim", "phone-video", "smolvla", "panda"]
    card_kw = dict(dataset_description=text, url=GITHUB)

    if args.dry_run:
        ds = LeRobotDataset(args.repo_id, root=root)
        card = create_lerobot_dataset_card(tags=tags, dataset_info=ds.meta.info, license=args.license,
                                           repo_id=args.repo_id, **card_kw)
        out = root.parent / f"{root.name}_README_preview.md"
        out.write_text(str(card))
        print(f"dry run: {info['total_episodes']} episodes, {info['total_frames']} frames; card -> {out}")
        return
    ds = LeRobotDataset(args.repo_id, root=root)
    ds.push_to_hub(tags=tags, license=args.license, private=args.private, **card_kw)
    print(f"pushed: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
