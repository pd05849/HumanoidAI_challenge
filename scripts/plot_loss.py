r"""Training-loss curves of the fine-tunes, read from the lerobot-train logs.

    python scripts/plot_loss.py outputs/models/T/train.log outputs/models/P/train.log \
        outputs/models/Pall/train.log --names T P Pall --out docs/img/train_loss

The logs are the ones train/train.sh writes (outputs/train/<RUN>.log); each model repo on the
Hub keeps a copy as train.log. Writes <out>.png (loss against training step, one line per run),
<out>.csv (every logged value) and <out>.md (the table printed at the end: loss over the last
1000 steps before 1K/5K/10K/15K/20K, passes over the data, training time).

The loss is SmolVLA's flow-matching loss on each run's OWN training data, averaged over each
logging window (100 steps). It shows that training converged. It does not say which policy is
better: the runs learn from different data, so only the evaluation success rates compare them.
"""
import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import numpy as np

# categorical slots 1-3 of the validated palette (scripts/compare_demos.py uses the same teleop/phone pair)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
COLORS = {"T": SLOTS[0], "P": SLOTS[1], "Pall": SLOTS[2]}
LABELS = {"T": "T: 50 teleop demos", "P": "P: 50 phone demos", "Pall": "Pall: 153 phone demos"}
INK, INK2, GRID, SURFACE = "#1f1f1e", "#5f5e5a", "#e6e5e1", "#fcfcfb"
MARKS = [1000, 5000, 10000, 15000, 20000]

EVENT = re.compile(r"(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \S+ (?:step:(?P<step>\S+) .*?\bepch:(?P<ep>[\d.]+) "
                   r".*?\bloss:(?P<loss>[\d.eE+-]+)|Checkpoint policy after step (?P<ckpt>\d+))")
START_LINE = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) .*Start offline training")
LOG_FREQ = re.compile(r"'log_freq': (\d+)")


def _big(s):
    """Inverse of LeRobot's format_big_number (precision 0): '200' -> (200, 0), '5K' -> (5000, 500)."""
    mult = {"": 1, "K": 1e3, "M": 1e6}[s[-1] if s[-1] in "KM" else ""]
    num = float(s.rstrip("KM"))
    return num * mult, (0.5 * mult if mult > 1 else 0)


def read_log(path, log_freq=None):
    """Logged steps, losses, passes over the data and wall time of one lerobot-train log.

    LeRobot prints the step rounded to thousands ('step:5K' covers 4500-5499), so the step of the
    k-th logged line is k * log_freq. Two checks keep that count honest: the printed (rounded) step,
    and the exact step of every 'Checkpoint policy after step N' line, which follows the log line
    of step N.
    """
    text = Path(path).read_text(errors="replace")
    if log_freq is None:
        m = LOG_FREQ.search(text)
        log_freq = int(m.group(1)) if m else 100
    rows = []
    hint = "A restarted or resumed run in the same log? Split the log at the restart, or pass --log-freq."
    for m in EVENT.finditer(text):
        if m["ckpt"]:
            if len(rows) * log_freq != int(m["ckpt"]):
                raise SystemExit(f"{path}: checkpoint at step {m['ckpt']} follows logged line {len(rows)}, "
                                 f"i.e. step {len(rows) * log_freq} (log_freq {log_freq}). {hint}")
            continue
        rows.append(m)
        value, tol = _big(m["step"])
        if abs(value - len(rows) * log_freq) > tol:
            raise SystemExit(f"{path}: logged line {len(rows)} says step {m['step']}, expected "
                             f"{len(rows) * log_freq} (log_freq {log_freq}). {hint}")
    if not rows:
        raise SystemExit(f"{path}: no 'step:... loss:...' lines (is it a lerobot-train log?)")
    stamp = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    start = START_LINE.search(text)
    t0 = stamp(start.group(1)) if start else stamp(rows[0]["ts"])
    hours = (stamp(rows[-1]["ts"]) - t0).total_seconds() / 3600
    return dict(step=np.arange(1, len(rows) + 1) * log_freq, loss=np.array([float(r["loss"]) for r in rows]),
                epochs=float(rows[-1]["ep"]), hours=hours, log_freq=log_freq)


def window_mean(run, end, span=1000):
    sel = (run["step"] > end - span) & (run["step"] <= end)
    return float(run["loss"][sel].mean()) if sel.any() else float("nan")


def table(runs, labels):
    marks = [m for m in MARKS if any(r["step"][-1] >= m for r in runs.values())]
    head = ["run"] + [f"loss at {m // 1000}K" for m in marks] + ["passes over the data", "training time"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for n, r in runs.items():
        cells = [labels[n]] + [f"{window_mean(r, m):.3f}" if r["step"][-1] >= m else "" for m in marks]
        lines.append("| " + " | ".join(cells + [f"{r['epochs']:.1f}", f"{r['hours']:.1f} h"]) + " |")
    return "\n".join(lines)


def plot(runs, labels, colors, out, smooth):
    try:
        import matplotlib
    except ImportError:
        raise SystemExit("needs matplotlib: pip install matplotlib") from None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter

    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.subplots_adjust(left=0.1, right=0.74, top=0.76, bottom=0.12)
    ends = []
    for n, r in runs.items():
        x = r["step"] / 1000
        ax.plot(x, r["loss"], color=colors[n], lw=1, alpha=0.25, zorder=2)
        k = max(1, smooth // r["log_freq"])
        y = np.convolve(r["loss"], np.ones(k) / k, mode="valid") if len(r["loss"]) >= k else r["loss"]
        ax.plot(x[len(x) - len(y):], y, color=colors[n], lw=2, zorder=3, label=labels[n])
        end = window_mean(r, r["step"][-1], smooth)   # = the table's last column when smooth is 1000
        ends.append([float(np.log10(end)), n, x[-1], end])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(np.arange(0, max(r["step"][-1] for r in runs.values()) / 1000 + 1, 5))
    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel("loss (log scale)")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # direct labels at the line ends, nudged apart so they never overlap
    lo, hi = (np.log10(v) for v in ax.get_ylim())
    gap = 0.055 * (hi - lo)
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for ly, n, xe, ye in ends:
        ax.annotate(f"{labels[n]}  {ye:.3f}", xy=(xe, ye), xytext=(xe * 1.02 + 0.2, 10 ** ly),
                    textcoords="data", va="center", fontsize=10, color=INK, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=colors[n], lw=1, shrinkA=0, shrinkB=2))
    ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=INK)
    fig.text(0.1, 0.955, "Training loss of the three fine-tunes", fontsize=13, color=INK, ha="left", va="top")
    fig.text(0.1, 0.905, "SmolVLA's flow-matching loss on each run's own training data. Thick line: running mean over "
             f"{smooth} steps;\nfaint line: each {runs[next(iter(runs))]['log_freq']}-step window; number: mean of the "
             f"final {smooth} steps. Lower loss does not mean a better\nrobot: the runs learn from different data, so "
             "only the evaluation success rates compare them.",
             fontsize=9.5, color=INK2, ha="left", va="top", linespacing=1.4)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="lerobot-train logs (train.log of each model repo)")
    ap.add_argument("--names", nargs="+", help="run names in the same order (default: T P Pall for 3 logs)")
    ap.add_argument("--out", default="docs/img/train_loss", help="output prefix for .png, .csv and .md")
    ap.add_argument("--smooth", type=int, default=1000,
                    help="running-mean window of the thick lines, in steps (1000 matches the table)")
    ap.add_argument("--log-freq", type=int, help="steps between logged lines (default: read from the log, else 100)")
    args = ap.parse_args()

    names = args.names or (["T", "P", "Pall"] if len(args.logs) == 3 else [Path(p).parent.name for p in args.logs])
    if len(names) != len(args.logs):
        raise SystemExit("give one name per log")
    runs = {n: read_log(p, args.log_freq) for n, p in zip(names, args.logs)}
    labels = {n: LABELS.get(n, n) for n in names}
    spare = iter(c for c in SLOTS if c not in [COLORS.get(n) for n in names])
    colors = {n: COLORS.get(n) or next(spare) for n in names}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(runs, labels, colors, out.with_suffix(".png"), args.smooth)
    all_steps = sorted(set().union(*(r["step"].tolist() for r in runs.values())))
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + names)
        lookup = {n: dict(zip(r["step"].tolist(), r["loss"].tolist())) for n, r in runs.items()}
        for s in all_steps:
            w.writerow([s] + [lookup[n].get(s, "") for n in names])
    md = table(runs, labels)
    out.with_suffix(".md").write_text(md + "\n")
    print(md)
    print(f"\nwrote {out.with_suffix('.png')}, {out.with_suffix('.csv')}, {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
