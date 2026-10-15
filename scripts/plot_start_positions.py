"""Where the frying pan and the moka pot start, seen from above: LIBERO's 50 test layouts, its 50
human demos (run T) and the 153 phone-derived demos (run Pall). README section 3.

    python scripts/plot_start_positions.py                     # writes docs/img/start_positions.png

Reads results/start_layouts.csv (see scripts/analyze_results.py for where its rows come from).
The dashed squares are the start regions of KITCHEN_SCENE3's BDDL files (the same in LIBERO-10
task 2 and LIBERO-90 task 21), in the table's frame: x from -0.075 to -0.025 m and y from -0.275 to
-0.225 m for the pan, x from 0.025 to 0.075 m and y from -0.025 to 0.025 m for the moka pot.
"""
import argparse
import csv
from pathlib import Path

# categorical slots of scripts/plot_loss.py and scripts/compare_demos.py: teleop blue, phone orange
SETS = [("test_layout", "LIBERO's 50 test layouts", "#3a3a38"),
        ("teleop_demo", "LIBERO's 50 human demos (run T)", "#2a78d6"),
        ("phone_demo", "my 153 phone demos (run Pall)", "#eb6834")]
INK, INK2, GRID, SURFACE = "#1f1f1e", "#5f5e5a", "#e6e5e1", "#fcfcfb"
REGIONS = {"pan": (-0.075, -0.025, -0.275, -0.225), "moka": (0.025, 0.075, -0.025, 0.025)}  # x0, x1, y0, y1 (m)


def view(x_m, y_m):
    """Table frame (m) -> the plot (cm): sideways = -y, away from the robot = x."""
    return -100 * y_m, 100 * x_m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", default="results/start_layouts.csv")
    ap.add_argument("--out", default="docs/img/start_positions.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows = list(csv.DictReader(open(args.layouts)))
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.edgecolor": GRID,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 3, figsize=(12.13, 3.27), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.76, bottom=0.08, wspace=0.04)
    for ax, (src, title, color) in zip(axes, SETS):
        r = [x for x in rows if x["source"] == src]
        for obj, marker in (("pan", "o"), ("moka", "s")):
            x0, x1, y0, y1 = REGIONS[obj]
            (a, b), (c, d) = view(x0, y1), view(x1, y0)   # corners of the region in plot coordinates
            ax.add_patch(Rectangle((a, b), c - a, d - b, fill=False, ls="--", lw=1, ec=INK2, zorder=1))
            pts = [view(float(x[f"{obj}_x_m"]), float(x[f"{obj}_y_m"])) for x in r]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=26, marker=marker, color=color,
                       edgecolors=SURFACE, linewidths=0.8, zorder=3)
            label = "frying pan\nstart area" if obj == "pan" else "moka pot\nstart area"
            ax.text((a + c) / 2, d + 0.6, label, ha="center", va="bottom", fontsize=10.5, color=INK2)
        ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
        ax.set_xlim(-6, 31)
        ax.set_ylim(-10.5, 12)
        ax.set_aspect("equal")
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("away from robot (cm) →")
    fig.text(0.01, 0.97, "Where the frying pan (circles) and the moka pot (squares) start: the same small areas "
             "in all three sets", fontsize=13, color=INK, ha="left", va="top")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
