"""The numbers behind README section 4, recomputed from the per-try logs in results/.

    python scripts/analyze_results.py                 # results/ and results/start_layouts.csv
    python scripts/analyze_results.py --root eval_logs

Prints, per policy: the first object lifted on each instruction, how the frying-pan tries failed,
how often the other object was disturbed, when the stove was turned on and the first object
lifted, and the frying-pan success split by how far the moka pot starts from the pan. Then how
far apart the pan and the moka pot start in the test layouts, in LIBERO's human demos and in the
phone demos.

results/start_layouts.csv holds the start positions (x, y of the pan and the moka pot):
  test_layout  the 50 fixed test layouts of LIBERO-90 task 21, in LIBERO's order, which is the
               order in which lerobot-eval runs them, so row k is try k in pan_episodes.jsonl
  teleop_demo  the first state of each of LIBERO's 50 human demos (the LIBERO-90 HDF5 of run T)
  phone_demo   the layout of each of the 153 phone-derived demos (sim/make_demos.py records)
Two-sided Fisher exact tests need scipy (skipped without it).
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

POLICIES = ["base", "T", "P", "Pall"]
CONTROL_HZ = 20
MOVED_CM = 1.0  # the other object counts as disturbed above this
LIFT_CM = 3.0   # same threshold as scripts/eval_with_log.py


def kind(name):
    if not name:
        return "nothing"
    return "pan" if "frypan" in name else ("moka" if "moka" in name else "other")


def value(d, key):
    return next((v for k, v in d.items() if key in k), 0.0)


def fisher(a, n_a, b, n_b):
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        return float("nan")
    return fisher_exact([[a, n_a - a], [b, n_b - b]])[1]


def rank_test(x, y):
    """Two-sided Mann-Whitney U p-value (no cut-off needed); None without both groups, nan without scipy."""
    if not x or not y:
        return None
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return float("nan")
    return mannwhitneyu(x, y).pvalue


def fmt_p(p):
    return "n/a (needs both successes and failures)" if p is None else f"p = {p:.2g}"


def load(root, name, tag):
    path = Path(root) / name / f"{tag}_episodes.jsonl"
    return [json.loads(line) for line in open(path)] if path.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results")
    ap.add_argument("--layouts", default="results/start_layouts.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.layouts)))
    dist = {src: [float(r["center_dist_cm"]) for r in rows if r["source"] == src]
            for src in ("test_layout", "teleop_demo", "phone_demo")}
    test = dist["test_layout"]
    median = statistics.median(test)
    order = sorted(range(len(test)), key=lambda k: test[k])
    closer, farther = order[:len(order) // 2], order[len(order) // 2:]

    for tag, target, other in [("pan", "frypan", "moka"), ("moka", "moka", "frypan")]:
        print(f"\n== told '{'frying pan' if tag == 'pan' else 'moka pot'}'")
        for name in POLICIES:
            recs = load(args.root, name, tag)
            if recs is None:
                continue
            first = {k: 0 for k in ("pan", "moka", "nothing", "other")}
            for r in recs:
                first[kind(r["first_object_picked_up"])] += 1
            fails = [r for r in recs if not r["success"]]
            never = sum(value(r["max_lift_cm"], target) <= LIFT_CM for r in fails)
            disturbed = [r for r in recs if value(r["max_move_cm"], other) > MOVED_CM]
            stove = [r["stove_turned_on_at_step"] for r in recs if r["stove_turned_on_at_step"] is not None]
            lift = [r["picked_up_at_step"] for r in recs if r["picked_up_at_step"] is not None]
            med = lambda xs: f"{statistics.median(xs):.0f} ({statistics.median(xs) / CONTROL_HZ:.1f} s)" if xs else "-"
            print(f"{name:5s} success {sum(r['success'] for r in recs):2d}/{len(recs)} | first lifted: pan {first['pan']}, "
                  f"moka pot {first['moka']}, nothing {first['nothing']} | failures without lifting the target: "
                  f"{never}/{len(fails)} | other object moved > {MOVED_CM:g} cm: {len(disturbed)} tries "
                  f"({sum(not r['success'] for r in disturbed)} failed) | median step: stove on {med(stove)}, "
                  f"first lift {med(lift)}")
            nl = [r for r in fails if value(r["max_lift_cm"], target) <= LIFT_CM]
            if nl:
                print(f"      in the {len(nl)} failures without lifting the target: target not moved at all in "
                      f"{sum(value(r['max_move_cm'], target) == 0 for r in nl)}, moved > {MOVED_CM:g} cm in "
                      f"{sum(value(r['max_move_cm'], target) > MOVED_CM for r in nl)}; other object moved > "
                      f"{MOVED_CM:g} cm in {sum(value(r['max_move_cm'], other) > MOVED_CM for r in nl)}")

    print(f"\n== frying-pan success by how far the moka pot starts from the pan (test layouts, "
          f"centre to centre; closer half {test[closer[0]]:.1f}-{test[closer[-1]]:.1f} cm, farther half "
          f"{test[farther[0]]:.1f}-{test[farther[-1]]:.1f} cm)")
    for name in POLICIES:
        recs = load(args.root, name, "pan")
        if recs is None or len(recs) != len(test):
            continue
        ok = lambda idx: sum(recs[k]["success"] for k in idx)
        hit = lambda idx: sum(value(recs[k]["max_move_cm"], "moka") > MOVED_CM for k in idx)
        tight = [k for k in range(len(test)) if test[k] < 25]
        rest = [k for k in range(len(test)) if test[k] >= 25]
        mid = [k for k in range(len(test)) if 25 <= test[k] < median]
        fails = [k for k in range(len(test)) if not recs[k]["success"]]
        print(f"{name:5s} closer {ok(closer):2d}/{len(closer)}, farther {ok(farther):2d}/{len(farther)} "
              f"(Fisher p = {fisher(ok(closer), len(closer), ok(farther), len(farther)):.2g}) | moka pot moved: "
              f"closer {hit(closer)}, farther {hit(farther)} | under 25 cm {ok(tight)}/{len(tight)}, 25 cm and more "
              f"{ok(rest)}/{len(rest)} (p = {fisher(ok(tight), len(tight), ok(rest), len(rest)):.2g}), 25 cm to the "
              f"median {ok(mid)}/{len(mid)} | failures among the 6 tightest starts: "
              f"{sum(k in order[:6] for k in fails)} of {len(fails)} | rank test, start distance of successes vs failures (no cut-off): "
              f"{fmt_p(rank_test([test[k] for k in range(len(test)) if recs[k]['success']], [test[k] for k in range(len(test)) if not recs[k]['success']]))}")

    print(f"\n== start distance between the pan and the moka pot (centre to centre)")
    for src, label in [("test_layout", "test layouts"), ("teleop_demo", "LIBERO human demos (T)"),
                       ("phone_demo", "phone demos (Pall)")]:
        x = dist[src]
        print(f"{label:24s} n={len(x):3d} median {statistics.median(x):5.1f} cm | closer than 25 cm: "
              f"{sum(v < 25 for v in x):3d} ({sum(v < 25 for v in x) / len(x):4.0%}) | closer than the test median "
              f"({median:.1f} cm): {sum(v < median for v in x):3d} ({sum(v < median for v in x) / len(x):4.0%})")

    base_pan, base_moka = load(args.root, "base", "pan"), load(args.root, "base", "moka")
    if base_pan and base_moka:
        a = sum(kind(r["first_object_picked_up"]) == "moka" for r in base_pan)
        b = sum(kind(r["first_object_picked_up"]) == "moka" for r in base_moka)
        print(f"\n== unchanged model lifts the moka pot first: told 'frying pan' {a}/{len(base_pan)}, told 'moka pot' "
              f"{b}/{len(base_moka)} (Fisher p = {fisher(a, len(base_pan), b, len(base_moka)):.2g}); frying pan moved "
              f"> {MOVED_CM:g} cm when told 'frying pan': {sum(value(r['max_move_cm'], 'frypan') > MOVED_CM for r in base_pan)}")


if __name__ == "__main__":
    main()
