#!/usr/bin/env python
"""Table of the eval results written by train/eval.sh (eval_logs/NAME/{pan,moka}...).

Per policy and instruction: successes k/n with a 95% Wilson interval, stove turned on, and which
object the robot lifted first (frying pan / moka pot / other / nothing), from the per-episode log
of scripts/eval_with_log.py; the success count is cross-checked against lerobot-eval's
eval_info.json. With two or more policies, a two-sided Fisher exact test compares the success
counts of every pair of policies on the same instruction (needs scipy).

usage:
  python train/summarize_evals.py [eval_logs] [--names base T P] [--md eval_logs/summary.md]
"""
import argparse
import json
import math
from pathlib import Path

TAGS = {"pan": "frying pan (LIBERO-90 #21)", "moka": "moka pot (LIBERO-10 #2)"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def obj_label(name):
    if not name:
        return "nothing"
    n = name.lower()
    if "frypan" in n or "frying" in n or "pan" in n.split("_"):
        return "frying pan"
    if "moka" in n:
        return "moka pot"
    return "other"


def load(root: Path, name: str, tag: str):
    jl = root / name / f"{tag}_episodes.jsonl"
    info = root / name / tag / "eval_info.json"
    if not jl.exists() and not info.exists():
        return None
    recs = [json.loads(l) for l in open(jl)] if jl.exists() else []
    pc = None
    if info.exists():
        with open(info) as f:
            ov = json.load(f).get("overall", {})
        pc = ov.get("pc_success")
    n = len(recs)
    k = sum(r["success"] for r in recs)
    first = {}
    for r in recs:
        lab = obj_label(r.get("first_object_picked_up"))
        first[lab] = first.get(lab, 0) + 1
    meta = root / name / f"{tag}.meta.json"
    return dict(name=name, tag=tag, n=n, k=k, rate=k / n if n else float("nan"), ci=wilson(k, n),
                stove=sum(r.get("stove_turned_on_at_step") is not None for r in recs), first=first,
                pc_success_eval_info=pc, meta=json.load(open(meta)) if meta.exists() else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="eval_logs")
    ap.add_argument("--names", nargs="+", default=None, help="policies (folder names) in table order")
    ap.add_argument("--md", default=None, help="also write the table to this markdown file")
    args = ap.parse_args()
    root = Path(args.root)
    names = args.names or sorted(p.name for p in root.iterdir() if p.is_dir())
    rows = []
    for name in names:
        for tag in TAGS:
            r = load(root, name, tag)
            if r:
                rows.append(r)
    if not rows:
        raise SystemExit(f"no results under {root}")

    lines = ["| policy | instruction | success | 95% CI | stove on | first object lifted |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        fo = ", ".join(f"{k} {v}" for k, v in sorted(r["first"].items(), key=lambda x: -x[1]))
        lines.append(f"| {r['name']} | {TAGS[r['tag']]} | {r['k']}/{r['n']} ({100 * r['rate']:.0f}%) | "
                     f"{100 * r['ci'][0]:.0f}-{100 * r['ci'][1]:.0f}% | {r['stove']}/{r['n']} | {fo} |")
        if r["pc_success_eval_info"] is not None and r["n"] and abs(r["pc_success_eval_info"] - 100 * r["rate"]) > 0.5:
            lines.append(f"| | WARNING: eval_info.json says {r['pc_success_eval_info']:.1f}% | | | | |")
        if r["meta"] is None:
            lines.append(f"| | WARNING: no {r['tag']}.meta.json: this run did not finish (redo it with train/eval.sh) | | | | |")
        elif r["n"] != r["meta"].get("n_episodes"):
            lines.append(f"| | WARNING: {r['n']} episodes logged, {r['meta'].get('n_episodes')} requested | | | | |")
    if len({r["meta"].get("eval_res") for r in rows if r["meta"]}) > 1:
        lines.append("\nWARNING: the policies were evaluated with different EVAL_RES settings.")
    try:
        from itertools import combinations

        from scipy.stats import fisher_exact
        tests = []
        for tag in TAGS:
            rs = [r for r in rows if r["tag"] == tag and r["n"]]
            for a, b in combinations(rs, 2):
                _, p = fisher_exact([[a["k"], a["n"] - a["k"]], [b["k"], b["n"] - b["k"]]])
                tests.append(f"- {tag}: {a['name']} {a['k']}/{a['n']} vs {b['name']} {b['k']}/{b['n']}: "
                             f"two-sided Fisher exact p = {p:.3g}")
        if tests:
            lines += ["", *tests]
    except ImportError:
        pass
    text = "\n".join(lines)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")
        print(f"\nwritten {args.md}")


if __name__ == "__main__":
    main()
