"""Summarise the tracked takes: one figure with every pan path, and a quality table.

    python scripts/summarize_tracks.py tracks/ --out tracks/summary

Reads tracks/<take>/path.json (from track_pan.py) and writes <out>.png and <out>.md.
Each take gets quality flags; flagged takes are listed, not silently dropped.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GROUP_COL = {60: "#2a78d6", 30: "#eb6834"}          # categorical slots 1 and 2


def take_key(p):
    m = re.findall(r"\d+", p.name)
    return (int(m[0]) if m else 0, p.name)


def load(d):
    doc = json.loads((d / "path.json").read_text())
    m = doc["metadata"]
    ph, mt, det = m["phases"], m["metrics"], m["detection"]
    n = m["frames"]
    measured = n - det["not_measured"] - len(det["rejected_outliers"])
    row = dict(
        take=d.name, fps=int(round(m["fps"])), frames=n,
        measured_pct=100.0 * measured / n,
        dist_cm=100 * mt["start_end_distance_horizontal_m"],
        lift_cm=100 * mt["lift_max_above_start_smoothed_m"],
        yaw_deg=mt["yaw_change_deg"], max_tilt_deg=mt["max_tilt_deg"],
        carry_s=ph["carry_duration_s"], peak_speed_cm_s=100 * mt["peak_speed_m_s"],
        jitter_mm=max(mt["rest_jitter_start"]["pos_std_3d_mm"], mt["rest_jitter_end"]["pos_std_3d_mm"]),
        look_down_deg=m["camera"]["look_down_deg"],
        lift_off_s=ph["lift_off_s"], touchdown_s=ph["touchdown_s"],
        start_cm=[100 * v for v in mt["start_m"]], end_cm=[100 * v for v in mt["end_m"]],
    )
    meas = np.array([r["measured"] for r in doc["path"]])
    longest, run = 0, 0
    for v in meas:
        run = 0 if v else run + 1
        longest = max(longest, run)
    row["longest_gap_s"] = longest / m["fps"]
    flags = []
    if row["longest_gap_s"] > 0.5:
        flags.append(f"{row['longest_gap_s']:.1f} s gap filled by interpolation")
    if row["measured_pct"] < 95:
        flags.append(f"only {row['measured_pct']:.0f}% of frames measured")
    if not 30 <= row["dist_cm"] <= 65:
        flags.append(f"carry distance {row['dist_cm']:.0f} cm")
    if row["lift_cm"] < 8:
        flags.append(f"low lift {row['lift_cm']:.0f} cm")
    if abs(row["yaw_deg"]) > 60:
        flags.append(f"pan turned {row['yaw_deg']:+.0f} deg")
    if row["carry_s"] is None or row["lift_off_s"] is None or row["touchdown_s"] is None:
        flags.append("lift-off / touchdown not found")
    elif not 1.0 <= row["carry_s"] <= 15.0:
        flags.append(f"carry took {row['carry_s']:.1f} s")
    if row["peak_speed_cm_s"] > 100:
        flags.append(f"implausible peak speed {row['peak_speed_cm_s']:.0f} cm/s (tracking error)")
    if row["max_tilt_deg"] > 25:
        flags.append(f"max tilt {row['max_tilt_deg']:.0f} deg")
    if row["jitter_mm"] > 2.0:
        flags.append(f"rest jitter {row['jitter_mm']:.1f} mm")
    row["flags"] = flags
    t = np.array([r["t"] for r in doc["path_smoothed"]])
    P = np.array([[r["x"], r["y"], r["z"]] for r in doc["path_smoothed"]])
    return row, t, P


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dirs = sorted([d for d in Path(args.tracks).iterdir() if (d / "path.json").exists()], key=take_key)
    if not dirs:
        raise SystemExit("no tracks found")
    data = [load(d) for d in dirs]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.patch.set_facecolor(SURFACE)
    for ax in (a1, a2):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # A: top view of every carry, world frame from the board
    a1.add_patch(plt.Rectangle((0, 0), 15, 21, fc="#efeeea", ec="#bdbcb6", lw=1))
    a1.text(7.5, 10.5, "board", ha="center", va="center", color=MUTED, fontsize=9)
    for row, t, P in data:
        c = GROUP_COL.get(row["fps"], MUTED)
        i0 = int(np.searchsorted(t, (row["lift_off_s"] or t[0]) - 0.5))
        i1 = int(np.searchsorted(t, (row["touchdown_s"] or t[-1]) + 0.5))
        a1.plot(100 * P[i0:i1, 0], 100 * P[i0:i1, 1], color=c, lw=1.6, alpha=0.75, solid_capstyle="round")
        a1.plot(*row["start_cm"][:2], "o", ms=6, mfc=c, mec=SURFACE, mew=1.5)
        a1.plot(*row["end_cm"][:2], "s", ms=6, mfc=c, mec=SURFACE, mew=1.5)
    a1.set_aspect("equal")
    a1.set_xlabel("x (cm): along the counter, right is toward the pan's start")
    a1.set_ylabel("y (cm): toward the wall")
    a1.set_title("Top view of every carry (circle = start, square = end)", loc="left", fontsize=11)

    # B: height above the start vs time, aligned at lift-off
    for row, t, P in data:
        if row["lift_off_s"] is None:
            continue
        c = GROUP_COL.get(row["fps"], MUTED)
        z = 100 * (P[:, 2] - row["start_cm"][2] / 100)
        sel = (t > row["lift_off_s"] - 0.5) & (t < (row["touchdown_s"] or t[-1]) + 1.0)
        a2.plot(t[sel] - row["lift_off_s"], z[sel], color=c, lw=1.6, alpha=0.75)
    a2.axhline(0, color=MUTED, lw=1)
    a2.set_xlabel("time since lift-off (s)")
    a2.set_ylabel("height above the start (cm)")
    a2.set_title("Lift profile of every carry", loc="left", fontsize=11)
    handles = [plt.Line2D([], [], color=c, lw=2.5, label=f"{f} fps takes") for f, c in GROUP_COL.items()
               if any(r["fps"] == f for r, _, _ in data)]
    a2.legend(handles=handles, frameon=False, loc="upper right", labelcolor=INK)
    fig.suptitle(f"{len(data)} tracked takes", x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=110, facecolor=SURFACE)

    # table
    hdr = ("| take | fps | measured | carry dist | lift | turn | max tilt | carry time | peak speed | "
           "rest jitter | flags |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for row, _, _ in data:
        lines.append(
            f"| {row['take']} | {row['fps']} | {row['measured_pct']:.0f}% | {row['dist_cm']:.1f} cm | "
            f"{row['lift_cm']:.1f} cm | {row['yaw_deg']:+.0f}° | {row['max_tilt_deg']:.0f}° | "
            f"{(row['carry_s'] or float('nan')):.1f} s | {row['peak_speed_cm_s']:.0f} cm/s | "
            f"{row['jitter_mm']:.2f} mm | {'; '.join(row['flags']) or 'ok'} |")
    rows = [r for r, _, _ in data]
    summ = (f"\n{len(rows)} takes; {sum(not r['flags'] for r in rows)} without flags. "
            f"Median carry {np.median([r['dist_cm'] for r in rows]):.1f} cm, "
            f"lift {np.median([r['lift_cm'] for r in rows]):.1f} cm, "
            f"turn {np.median([r['yaw_deg'] for r in rows]):+.0f}°, "
            f"carry time {np.nanmedian([r['carry_s'] or np.nan for r in rows]):.1f} s.\n")
    Path(f"{args.out}.md").write_text(hdr + "\n".join(lines) + "\n" + summ)
    print(hdr + "\n".join(lines) + summ)
    print(f"wrote {args.out}.png, {args.out}.md")


if __name__ == "__main__":
    main()
