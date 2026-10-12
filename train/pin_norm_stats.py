#!/usr/bin/env python
"""Pin a dataset's state/action normalization stats to the ones the pretrained policy uses.

Why: when lerobot-train fine-tunes from --policy.path, it REPLACES the policy's normalization
stats with the training dataset's meta/stats.json (lerobot_train.py: preprocessor_overrides
"normalizer_processor": {"stats": dataset.meta.stats}). Runs P and T would then each normalize
state and action with different statistics (scripted phone demos vs human teleop), both different
from what smolvla_libero was trained with. Writing the pretrained policy's own mean/std into
both merged datasets keeps the normalization identical across runs and equal to pretraining, so
the only difference between P and T is the frying-pan demos.

The stats are read from the policy's preprocessor (policy_preprocessor.json -> the
normalizer_processor step's .safetensors), on the Hub or in a local checkpoint folder.
Only --keys are replaced (default observation.state and action; SmolVLA normalizes images with
IDENTITY). The original file is kept as meta/stats.dataset.json; --restore puts it back.

usage:
  python train/pin_norm_stats.py --dataset outputs/datasets/P [--policy HuggingFaceVLA/smolvla_libero]
  python train/pin_norm_stats.py --dataset outputs/datasets/P --restore
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import write_json  # noqa: E402


def _fetch(policy: str, filename: str) -> Path:
    p = Path(policy)
    if p.is_dir():
        f = p / filename
        if not f.is_file():
            raise SystemExit(f"{f} not found")
        return f
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(policy, filename))


def policy_norm_stats(policy: str, config_filename: str = "policy_preprocessor.json") -> dict:
    """({feature: {stat: np.ndarray}}, norm_map, state file name) from the policy's
    normalizer_processor step."""
    from safetensors.numpy import load_file
    cfg_path = _fetch(policy, config_filename)
    with open(cfg_path) as f:
        cfg = json.load(f)
    steps = [s for s in cfg.get("steps", []) if s.get("registry_name") == "normalizer_processor"]
    if not steps or not steps[0].get("state_file"):
        raise SystemExit(f"{policy}: no normalizer_processor step with a state_file in {config_filename}; "
                         f"steps: {[s.get('registry_name') for s in cfg.get('steps', [])]}")
    flat = load_file(str(_fetch(policy, steps[0]["state_file"])))
    out = {}
    for k, v in flat.items():
        feat, stat = k.rsplit(".", 1)
        out.setdefault(feat, {})[stat] = np.asarray(v, dtype=np.float64)
    norm_map = (steps[0].get("config") or {}).get("norm_map")
    return out, norm_map, steps[0]["state_file"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="LeRobot dataset folder whose meta/stats.json is edited")
    ap.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero", help="Hub repo id or local checkpoint dir")
    ap.add_argument("--keys", nargs="+", default=["observation.state", "action"])
    ap.add_argument("--restore", action="store_true", help="put the dataset's own stats back")
    args = ap.parse_args()

    meta = Path(args.dataset) / "meta"
    stats_p, backup = meta / "stats.json", meta / "stats.dataset.json"
    if args.restore:
        if not backup.exists():
            raise SystemExit(f"no backup {backup}")
        shutil.copy(backup, stats_p)
        (meta / "stats.pinned.json").unlink(missing_ok=True)  # train.sh warns again: stats not pinned
        print(f"restored {stats_p} from {backup}")
        return
    if not stats_p.exists():
        raise SystemExit(f"{stats_p} not found")
    if not backup.exists():
        shutil.copy(stats_p, backup)
    with open(backup) as f:
        ds_stats = json.load(f)               # always start from the dataset's own stats

    pol, norm_map, src = policy_norm_stats(args.policy)
    print(f"policy {args.policy}: normalizer stats for {sorted(pol)} ({src}); norm_map {norm_map}")
    new = json.loads(json.dumps(ds_stats))
    for key in args.keys:
        if key not in pol:
            raise SystemExit(f"the policy has no stats for {key!r} (has {sorted(pol)})")
        if key not in ds_stats:
            raise SystemExit(f"the dataset has no stats for {key!r}")
        for stat, v in pol[key].items():
            if stat in ds_stats[key] and np.asarray(ds_stats[key][stat]).shape != v.shape:
                raise SystemExit(f"{key}.{stat}: policy shape {v.shape} vs dataset {np.asarray(ds_stats[key][stat]).shape}")
            if stat == "count":
                continue
            new[key][stat] = v.tolist()
        np.set_printoptions(precision=4, suppress=True, linewidth=150)
        for stat in ("mean", "std"):
            if stat in pol[key]:
                print(f"  {key}.{stat}\n     policy  {pol[key][stat]}\n     dataset {np.asarray(ds_stats[key][stat])}")
    write_json(stats_p, new)
    write_json(meta / "stats.pinned.json", {"policy": args.policy, "keys": args.keys, "state_file": src})
    print(f"wrote {stats_p} (dataset's own stats kept in {backup})")


if __name__ == "__main__":
    main()
