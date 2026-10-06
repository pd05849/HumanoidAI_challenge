"""Summarise sim/make_demos.py runs: how many layouts were tried and kept, why the others were
dropped, and how closely the kept demos follow the phone paths.

    python sim/summarize_demos.py demos/r0 demos/r1 demos/r2 [--md demos/summary.md]
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np


def load(dirs):
    rows = []
    for d in dirs:
        for sub in ("episodes", "failed"):
            for f in glob.glob(os.path.join(d, sub, "*.json")):
                with open(f) as fh:
                    m = json.load(fh)
                if m.get("generator_version") is None:
                    continue
                m["_file"] = f
                rows.append(m)
    return rows


def reason(m):
    if m.get("kept"):
        return "kept"
    if "traceback" in m:
        return "error"
    if not m.get("success"):
        return "task failed"
    s = str(m.get("status", "")) + " " + str(m.get("not_kept_reason", ""))
    if m.get("moka", {}).get("disturbed"):
        return "moka pot disturbed"
    if "tracking" in s or (m.get("metrics", {}).get("rms_sync_3d_cm", 0) or 0) > 2.0:
        return "path error > 2 cm"
    return "other: " + s.strip()[:60]


def q(v):
    v = np.asarray([x for x in v if x is not None], float)
    if not len(v):
        return "-"
    return f"median {np.median(v):.2f}, range {v.min():.2f}-{v.max():.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    rows = load(args.dirs)
    kept = [m for m in rows if m.get("kept")]
    reasons = Counter(reason(m) for m in rows)
    takes = Counter(m["take"] for m in kept)
    info = lambda k: [m.get("info", {}).get(k) for m in kept]
    met = lambda k: [m.get("metrics", {}).get(k) for m in kept]
    raised = [m for m in kept if (m.get("info", {}).get("raise_max_cm") or 0) > 1.0]
    infeasible = [m for m in rows if m.get("info", {}).get("grasp_choice", {}).get("feasible") is False]
    lines = [
        f"layouts tried: {len(rows)}; kept: {len(kept)} ({100 * len(kept) / max(1, len(rows)):.0f} %)",
        "not kept: " + ", ".join(f"{k} {v}" for k, v in reasons.items() if k != "kept"),
        f"kept per take: {min(takes.values()) if takes else 0}-{max(takes.values()) if takes else 0} "
        f"({len(takes)} takes)",
        f"no collision-free grasp: {len(infeasible)} tried, {sum(m.get('kept', False) for m in infeasible)} kept",
        f"path following, RMS (cm): {q(met('rms_sync_3d_cm'))}",
        f"path following, max (cm): {q(met('max_sync_3d_cm'))}",
        f"path raised over the moka pot (> 1 cm): {len(raised)} of {len(kept)} kept, "
        f"raise {q([m['info']['raise_max_cm'] for m in raised])} cm",
        f"pan turning in the grasp (deg): {q(info('grasp_slip_deg'))}",
        f"first success at control step: {q([m.get('first_success_step') for m in kept])} (eval limit 400)",
        f"path scale: {q(info('scale'))}; rotation (deg): {q(info('rotation_deg'))}",
        f"sim pan peak height above table (cm): {q(met('pan_peak_z_cm'))}",
    ]
    ed = [m.get("eval_distance", {}).get("min_joint_cm") for m in kept]
    if any(v is not None for v in ed):
        lines.append(f"distance to the nearest of the 50 test layouts (pan + moka, cm): {q(ed)}")
    text = "\n".join(lines)
    print(text)
    if args.md:
        with open(args.md, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
