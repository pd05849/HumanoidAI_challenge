r"""Side-by-side comparison of two LIBERO-format LeRobot datasets of the same task: LIBERO's
human-teleoperated demos (run T) and the phone-derived demos (run P).

    hf download Delcastillo8/libero90_frying_pan_phone_50 --repo-type dataset --local-dir datasets/phone_pan_P
    python scripts/compare_demos.py --teleop outputs/datasets/teleop_pan \
        --phone datasets/phone_pan_P --out docs/img/teleop_vs_phone

Writes
  <out>.gif / <out>.mp4   one demo of each, side by side, front and wrist cameras exactly as the model
                          sees them (LIBERO's images are mirrored left-right), at the 20 Hz control rate:
                          one dataset frame = one control step, whatever the dataset's fps label says
  <out>_paths.png         every demo's gripper path: top view and height over time, one column per dataset
  <out>.md                caption + summary table (duration, carry, speed, smoothness, gripper use)

Both converters leave out no-op steps (arm still, gripper unchanged): the phone demos have almost
none, but convert_teleop_hdf5.py skips the human operator's pauses, so teleop times are active time.

Episodes for the video: --teleop-ep / --phone-ep (dataset episode index), by default the one whose
length is closest to the dataset's median. Needs the LeRobot environment (lerobot, matplotlib,
opencv); ffmpeg for the gif colours and the mp4 (without it, a plainer gif is written with Pillow).
"""
import argparse
import glob
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from common import KEY_STATE, camera_keys, read_info  # noqa: E402

CONTROL_HZ = 20                      # LIBERO / robosuite control frequency
HOLD_GAP_M = 0.01                    # finger gap above which a closed gripper is holding something
FULL_CMD = 0.93                      # LIBERO's SpaceMouse translation command tops out at 0.9375
COLORS = {"teleop": "#2a78d6", "phone": "#eb6834"}   # validated categorical pair (dataviz)
INK, INK2, GRID, SURFACE = "#1f1f1e", "#5f5e5a", "#e6e5e1", "#fcfcfb"


# ----------------------------------------------------------------------------- data

def load_table(root):
    """All frames of a dataset (parquet only, no video): episode, frame, state, action."""
    import pandas as pd
    files = sorted(glob.glob(str(Path(root) / "data" / "*" / "*.parquet")))
    if not files:
        raise SystemExit(f"{root}: no data/*/*.parquet (not a LeRobot v3 dataset?)")
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_values(["episode_index", "frame_index"])
    eps = {}
    for ep, e in df.groupby("episode_index"):
        eps[int(ep)] = {"state": np.stack(e[KEY_STATE].to_numpy()).astype(np.float64),
                        "action": np.stack(e["action"].to_numpy()).astype(np.float64)}
    return eps


def closed_segments(action):
    """[start, end) index ranges where the gripper command is 'close' (action[6] > 0)."""
    g = np.r_[False, action[:, 6] > 0, False].astype(int)
    d = np.diff(g)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def carry_segment(ep):
    """The pan carry: among the closed-gripper stretches whose fingers stopped on something (finger
    gap state[6]-state[7] above 1 cm; shut on nothing, e.g. pushing the knob, it reads ~1 mm), the
    one that moves the gripper furthest over the table. Falls back to all closed stretches."""
    s = ep["state"]
    p, gap = s[:, :3], s[:, 6] - s[:, 7]
    segs = closed_segments(ep["action"])
    held = [se for se in segs if np.median(gap[se[0]:se[1]]) > HOLD_GAP_M] or segs
    if not held:
        return None
    return max(held, key=lambda se: np.linalg.norm(p[se[1] - 1, :2] - p[se[0], :2]))


def episode_stats(ep):
    s, a = ep["state"], ep["action"]
    p = s[:, :3]                                       # gripper (end-effector) position, m
    t = len(p) / CONTROL_HZ
    step = np.linalg.norm(np.diff(p, axis=0), axis=1) if len(p) > 1 else np.zeros(1)
    speed = step * CONTROL_HZ
    moving = speed > 0.01                              # > 1 cm/s
    # translation only: LIBERO's SpaceMouse scales rotation differently (max 0.25 vs 1 here)
    da = np.abs(np.diff(a[:, :3], axis=0)).mean(axis=1) if len(a) > 1 else np.zeros(1)
    seg = carry_segment(ep)
    if seg is None:                                    # never grasped: no carry, not "the whole demo"
        carry = {"grasp_s": np.nan, "carry_s": np.nan, "carry_cm": np.nan, "carry_lift_cm": np.nan}
    else:
        c0, c1 = seg
        carry = {"grasp_s": c0 / CONTROL_HZ,
                 "carry_s": (c1 - c0) / CONTROL_HZ,
                 "carry_cm": 100 * np.linalg.norm(p[c1 - 1, :2] - p[c0, :2]),
                 "carry_lift_cm": 100 * (p[c0:c1, 2].max() - p[c0, 2])}
    return {
        "duration_s": t,
        "path_m": step.sum(),
        **carry,
        "n_closed": len(closed_segments(a)),
        "speed_cms": 100 * speed[moving].mean() if moving.any() else 0.0,
        "peak_speed_cms": 100 * np.percentile(speed, 95),
        "action_change": da.mean(),
        "max_cmd": np.abs(a[:, :3]).max(),
        "saturated": np.mean(np.any(np.abs(a[:, :3]) >= FULL_CMD, axis=1)),
        "gripper_closed": np.mean(a[:, 6] > 0),
    }


def median_episode(eps):
    lens = {k: len(v["state"]) for k, v in eps.items()}
    med = np.median(list(lens.values()))
    return min(lens, key=lambda k: (abs(lens[k] - med), k))


def decode_episode(root, ep):
    """(front, wrist) uint8 frames (T, H, W, 3) of one episode, as stored (= as the model sees them)."""
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.video_utils import decode_video_frames
    root = Path(root)
    info = read_info(root)
    fps = float(info["fps"])
    cam1, cam2 = camera_keys(info["features"])
    meta = LeRobotDatasetMetadata(f"local/{root.name}", root=root)
    files = sorted(glob.glob(str(root / "data" / "*" / "*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index"]) for f in files])
    n = int((df["episode_index"] == ep).sum())
    epm = meta.episodes[ep]
    out = []
    for key in (cam1, cam2):
        start = float(epm[f"videos/{key}/from_timestamp"])
        ts = [start + i / fps for i in range(n)]
        fr = decode_video_frames(root / meta.get_video_file_path(ep, key), ts, tolerance_s=0.5 / fps,
                                 return_uint8=True)
        out.append(fr.permute(0, 2, 3, 1).cpu().numpy())
    return out


# ----------------------------------------------------------------------------- video

def _font():
    """Regular and bold TrueType fonts (DejaVu, Arial or Liberation), else PIL's default."""
    from PIL import ImageFont
    import matplotlib
    mpl = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    cands = [(str(mpl / "DejaVuSans.ttf"), str(mpl / "DejaVuSans-Bold.ttf")), ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
             ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
             ("arial.ttf", "arialbd.ttf"), ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
             ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
             ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")]
    for reg, bold in cands:
        try:
            ImageFont.truetype(reg, 12)
            ImageFont.truetype(bold, 12)
            return lambda size, b=False, reg=reg, bold=bold: ImageFont.truetype(bold if b else reg, size)
        except OSError:
            continue
    return lambda size, b=False: ImageFont.load_default(size)   # Pillow >= 10.1


def _text(img, xy, s, size, color, bold=False):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text(xy, s, font=FONT(size, bold), fill=color)
    img[:] = np.asarray(im)


def _demos(k):
    return f"{k} demo" + ("" if k == 1 else "s")


def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _track(g, carry, T, n, pw, col, h=14):
    """Gripper timeline of one demo, drawn once: closed = dataset colour, open = grey, after the demo
    ends = nothing (surface), so 'ended' never looks like 'gripper open'. The carry is labelled."""
    from PIL import Image, ImageDraw
    track = np.empty((h, pw, 3), np.uint8)
    track[:] = _hex(SURFACE)
    xe = int(round(T * pw / n))
    track[:, :xe] = _hex(GRID)
    for t in np.flatnonzero(g):
        track[:, int(t * pw / n):max(int(t * pw / n) + 1, int((t + 1) * pw / n))] = col
    if carry is not None:
        im = Image.fromarray(track)
        d, f = ImageDraw.Draw(im), FONT(11, True)
        x0, x1 = int(carry[0] * pw / n), int(carry[1] * pw / n)
        w = d.textlength("pan in hand", font=f)
        if w + 8 <= x1 - x0:                                 # only if it fits with padding
            d.text(((x0 + x1 - w) / 2, 0), "pan in hand", font=f, fill=(255, 255, 255))
        track = np.asarray(im).copy()
    return track


FRONT_LABEL = "front camera (mirrored)"


def make_frames(sides, instruction, big=256, small=128, gap=12):
    """sides: list of dicts {label, sub, color, front, wrist, gripper_closed, carry}. Yields RGB frames."""
    n = max(len(s["front"]) for s in sides)
    pw = big + 6 + small                                # panel width
    W = len(sides) * pw + (len(sides) + 1) * gap
    H_TOP, H_HEAD, H_BAR = 40, 50, 44
    H = H_TOP + H_HEAD + big + H_BAR + 8
    surf = _hex(SURFACE)
    tracks = [_track(s["gripper_closed"], s.get("carry"), len(s["front"]), n, pw, _hex(s["color"]))
              for s in sides]
    for i in range(n):
        img = np.empty((H, W, 3), np.uint8)
        img[:] = surf
        _text(img, (gap, 11), f"\u201c{instruction}\u201d", 17, _hex(INK))
        _text(img, (W - gap - 150, 13), f"t = {i / CONTROL_HZ:4.1f} s", 15, _hex(INK2))
        for k, s in enumerate(sides):
            x0 = gap + k * (pw + gap)
            y0 = H_TOP
            col = _hex(s["color"])
            img[y0 + 4:y0 + 8, x0:x0 + pw] = col               # colour key: a bar, not coloured text
            _text(img, (x0, y0 + 12), s["label"], 16, _hex(INK), bold=True)
            _text(img, (x0, y0 + 31), s["sub"], 13, _hex(INK2))
            T = len(s["front"])
            j = min(i, T - 1)
            y1 = y0 + H_HEAD
            img[y1:y1 + big, x0:x0 + big] = cv2.resize(s["front"][j], (big, big), interpolation=cv2.INTER_CUBIC)
            img[y1:y1 + small, x0 + big + 6:x0 + pw] = cv2.resize(s["wrist"][j], (small, small),
                                                                 interpolation=cv2.INTER_AREA)
            _text(img, (x0 + big + 6, y1 + small + 4), "wrist camera", 12, _hex(INK2))
            if i >= T:                                          # finished: dim a little, keep the end state visible
                for xa, sz in ((x0, big), (x0 + big + 6, small)):
                    img[y1:y1 + sz, xa:xa + sz] = (0.7 * img[y1:y1 + sz, xa:xa + sz] + 0.3 * 255).astype(np.uint8)
                img[y1 + big - 28:y1 + big, x0:x0 + big] = _hex(INK)
                _text(img, (x0 + 8, y1 + big - 23), f"done: demo ended at {T / CONTROL_HZ:.1f} s", 14,
                      (250, 250, 250), bold=True)
            lw = min(big, int(FONT(12).getlength(FRONT_LABEL)) + 12)            # label backdrop
            img[y1:y1 + 20, x0:x0 + lw] = (0.45 * img[y1:y1 + 20, x0:x0 + lw]).astype(np.uint8)
            _text(img, (x0 + 6, y1 + 3), FRONT_LABEL, 12, (250, 250, 250))
            yb = y1 + big + 10
            img[yb:yb + 14, x0:x0 + pw] = tracks[k]
            xp = x0 + min(int(j * pw / n), pw - 1)
            img[yb - 6:yb, max(xp - 1, x0):xp + 1] = _hex(INK)      # playhead above and below the bar,
            img[yb + 14:yb + 18, max(xp - 1, x0):xp + 1] = _hex(INK)  # never through its label
            _text(img, (x0, yb + 18), "coloured = gripper closed,  black tick = now", 12, _hex(INK2))
        yield img


def write_media(frames, out, gif_fps=10):
    """mp4 at 20 fps (real time) and a gif at gif_fps (every 20/gif_fps-th frame, still real time)."""
    frames = list(frames)
    skip = max(1, round(CONTROL_HZ / gif_fps))
    tmp = Path(tempfile.mkdtemp())
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(str(tmp / f"f{i:04d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        def ffmpeg(*a):
            if shutil.which("ffmpeg") is None:
                return False
            cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", str(CONTROL_HZ), "-i", str(tmp / "f%04d.png"), *a]
            return subprocess.run(cmd).returncode == 0

        if not ffmpeg("-vf", f"select='not(mod(n\\,{skip}))',setpts=N/({CONTROL_HZ / skip}*TB),"
                             "split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];"
                             "[b][p]paletteuse=dither=none:diff_mode=rectangle",
                      "-r", str(CONTROL_HZ / skip), "-loop", "0", f"{out}.gif"):
            from PIL import Image
            ims = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames[::skip]]
            ims[0].save(f"{out}.gif", save_all=True, append_images=ims[1:], duration=int(1000 * skip / CONTROL_HZ),
                        loop=0)
        mp4 = ["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", f"{out}.mp4"]
        if not (ffmpeg("-c:v", "libx264", "-crf", "20", *mp4) or ffmpeg("-c:v", "mpeg4", "-q:v", "2", *mp4)):
            print("no mp4 written (ffmpeg missing or without an H.264/MPEG-4 encoder); the gif is complete")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- figure

def landmarks(eps):
    """Median gripper position at the knob turn, the grasp and the release, from demos whose first
    closed-gripper stretch is not the carry (the phone demos: scripted, knob first)."""
    knob, grasp, rel = [], [], []
    for e in eps.values():
        p, segs, c = e["state"][:, :3], closed_segments(e["action"]), carry_segment(e)
        if c is None:
            continue
        grasp.append(p[c[0], :2])
        rel.append(p[c[1] - 1, :2])
        first = segs[0]
        if first != c:
            k = first[0] + int(np.argmin(p[first[0]:first[1], 2]))
            knob.append(p[k, :2])
    out = {}
    for name, v in (("knob", knob), ("grasp", grasp), ("let go", rel)):
        if len(v) >= max(3, len(eps) // 2):
            out[name] = np.median(np.array(v), axis=0)
    return out


def make_figure(data, out_png, labels, notes, highlight, landmark_source="phone"):
    """notes[n]: one line under each column title; highlight[n]: the episode shown in the video."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    names = list(data)
    fig, ax = plt.subplots(2, len(names), figsize=(5.2 * len(names), 8.8), squeeze=False,
                           gridspec_kw={"height_ratios": [1.15, 1]})
    allp = np.concatenate([e["state"][:, :3] for n in names for e in data[n].values()])
    # top view in cm from the (shared) start position, height in cm above the lowest gripper point
    o = np.median([e["state"][0, :3] for n in names for e in data[n].values()], axis=0)
    z0 = allp[:, 2].min()
    xy = lambda p: (-100 * (p[..., 1] - o[1]), 100 * (p[..., 0] - o[0]))
    tmax = max(len(e["state"]) for n in names for e in data[n].values()) / CONTROL_HZ
    marks = landmarks(data[landmark_source]) if landmark_source in data else {}
    X, Y = xy(allp)
    for c, n in enumerate(names):
        col = COLORS[n]
        a, b = ax[0, c], ax[1, c]
        for m in names:                                   # the other dataset: grey context underneath
            if m == n:
                continue
            for e in data[m].values():
                p = e["state"][:, :3]
                b.plot(np.arange(len(p)) / CONTROL_HZ, 100 * (p[:, 2] - z0), color="#c9c8c3", lw=0.8,
                       alpha=0.35, zorder=1)
        for k, e in data[n].items():
            p = e["state"][:, :3]
            hi = k == highlight.get(n)
            kw = dict(color=col, lw=2.2 if hi else 0.9, alpha=1.0 if hi else 0.3, zorder=4 if hi else 2,
                      solid_capstyle="round")
            a.plot(*xy(p), **kw)
            a.plot(*xy(p[0]), "o", ms=3.5, color=col, mec=SURFACE, mew=0.8, alpha=0.8, zorder=3)
            b.plot(np.arange(len(p)) / CONTROL_HZ, 100 * (p[:, 2] - z0), **kw)
        for name, v in marks.items():               # scene landmarks, same on every panel
            mx, my = xy(np.r_[v, 0.0])
            a.plot(mx, my, "o", ms=13, mfc="none", mec=INK, mew=1.1, zorder=5)
            a.annotate(name, (mx, my), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=10,
                       color=INK, zorder=6)
        a.set_title(f"{labels[n]}: {_demos(len(data[n]))}", loc="left", fontsize=12, color=INK, pad=22,
                    fontweight="bold")
        a.text(0, 1.025, notes[n], transform=a.transAxes, fontsize=10, color=INK2, va="bottom")
        a.set_xlim(X.min() - 3, X.max() + 3)
        a.set_ylim(Y.min() - 3, Y.max() + 3)
        a.set_aspect("equal")
        a.set_xlabel("\u2190 robot's left      sideways (cm)      robot's right \u2192")
        b.set_xlim(0, tmax)
        b.set_ylim(-2, 100 * (allp[:, 2].max() - z0) + 2)
        b.set_xlabel("time (s)")
        if c == 0:
            a.set_ylabel("away from the robot (cm) \u2192")
            b.set_ylabel("gripper height (cm above its lowest point)")
        for x in (a, b):
            x.grid(True, color=GRID, lw=0.6)
            x.set_axisbelow(True)
            for s in x.spines.values():
                s.set_visible(False)
    import textwrap
    H = fig.get_figheight()
    fig.text(0.015, 1 - 0.22 / H, "Where the gripper goes (top view, from its start position) and how high, over time",
             fontsize=13, color=INK, ha="left", va="top")
    ring = (f" Rings: where the {labels[landmark_source]} demos turn the knob, grasp the handle and let go."
            if marks else "")
    sub = textwrap.fill("One line per demo, dot = start; the bold line is the demo in the video; in the height panels, "
                        "grey = the other dataset." + ring, width=62 * len(names))
    fig.text(0.015, 1 - 0.52 / H, sub, fontsize=10, color=INK2, ha="left", va="top", linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.62 + 0.2 * sub.count("\n")) / H))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def stats_table(data, labels):
    rows = [("demo length (s)", "duration_s", "{:.1f}"),
            ("gripper travel, whole demo (m)", "path_m", "{:.2f}"),
            ("pan grasped at (s from the start)", "grasp_s", "{:.1f}"),
            ("pan in hand (s)", "carry_s", "{:.1f}"),
            ("pan moved, grasp to release, straight line (cm)", "carry_cm", "{:.0f}"),
            ("pan lifted: highest point above the grasp (cm)", "carry_lift_cm", "{:.0f}"),
            ("gripper speed while moving, mean (cm/s)", "speed_cms", "{:.1f}"),
            ("gripper speed, fastest 5 % of steps (cm/s)", "peak_speed_cms", "{:.0f}"),
            ("translation command change per step, mean (lower = smoother)", "action_change", "{:.3f}"),
            ("largest translation command (limit 1; LIBERO's 3-D mouse reaches 0.94)", "max_cmd", "{:.2f}"),
            (f"steps at full translation speed (command \u2265 {FULL_CMD})", "saturated", "{:.0%}"),
            ("steps with the gripper commanded closed", "gripper_closed", "{:.0%}"),
            ("closed-gripper stretches per demo (knob, carry, regrasps)", "n_closed", "{:.0f}")]
    st = {n: [episode_stats(e) for e in data[n].values()] for n in data}
    head = "| median (min\u2013max) | " + " | ".join(f"{labels[n]}, {_demos(len(st[n]))}" for n in data) + " |"
    lines = [head, "|---|" + "---|" * len(data)]
    for r in rows:
        name, key, fmt = r[:3]
        mul = r[3] if len(r) > 3 else 1
        cells = []
        for n in data:
            v = np.array([s[key] for s in st[n]], dtype=float) * mul
            v = v[np.isfinite(v)]
            miss = len(st[n]) - len(v)
            cells.append("n/a (no grasp)" if not len(v) else
                         f"{fmt.format(np.median(v))} ({fmt.format(v.min())}\u2013{fmt.format(v.max())})"
                         + (f" [{miss} without a grasp]" if miss else ""))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    note = (f"Median over each dataset's demos, (min\u2013max) in brackets. Times count {CONTROL_HZ} Hz control "
            "steps. No-op steps (arm still, gripper unchanged) are not in the datasets: the phone demos have "
            "almost none, but LIBERO's converter skips the operator's pauses, so teleop times are active time. "
            "\"Pan in hand\" is the longest closed-gripper stretch with the fingers stopped on something; a "
            "regrasp splits it.")
    return note + "\n\n" + "\n".join(lines)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teleop", required=True, help="LeRobot dataset of LIBERO's teleop demos (teleop_pan)")
    ap.add_argument("--phone", required=True, help="LeRobot dataset of phone-derived demos (phone_pan_P)")
    ap.add_argument("--out", default="docs/img/teleop_vs_phone", help="output prefix")
    ap.add_argument("--teleop-ep", type=int, default=None, help="dataset episode index (default: median length)")
    ap.add_argument("--phone-ep", type=int, default=None, help="dataset episode index (default: median length)")
    ap.add_argument("--teleop-label", default="LIBERO teleop")
    ap.add_argument("--phone-label", default="phone-derived")
    ap.add_argument("--gif-fps", type=float, default=10)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    roots = {"teleop": args.teleop, "phone": args.phone}
    labels = {"teleop": args.teleop_label, "phone": args.phone_label}
    subs = {"teleop": "a person steered the robot with a 3-D mouse (LIBERO)",
            "phone": "scripted knob turn and grasp; carry = my phone video"}
    data = {n: load_table(r) for n, r in roots.items()}
    tasks = {}
    for n, r in roots.items():
        import pandas as pd
        tasks[n] = list(pd.read_parquet(Path(r) / "meta" / "tasks.parquet").index)
    if tasks["teleop"] != tasks["phone"]:
        print(f"note: different task strings {tasks}")
    instruction = tasks["phone"][0]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    table = stats_table(data, labels)
    caption = (f"Left: one of the {labels['teleop']} demos (a person steering the robot). Right: one "
               f"{labels['phone']} demo (knob turn, grasp and release scripted; the carry follows the pan "
               "path tracked in a phone video). Same task, same simulator, the two cameras the policy sees "
               "(mirrored left-right, as LIBERO stores them), 20 control steps per second. The bar under each "
               "side shows when the gripper is closed; the black tick is now.")
    Path(f"{args.out}.md").write_text(caption + "\n\n" + table + "\n")
    print(table)
    eps = {"teleop": args.teleop_ep, "phone": args.phone_ep}
    eps = {n: eps[n] if eps[n] is not None else median_episode(data[n]) for n in data}
    notes = {"teleop": "a person steered the robot with a 3-D mouse",
             "phone": "knob turn + grasp scripted; carry = my phone video"}
    make_figure(data, f"{args.out}_paths.png", labels, notes, eps)
    print(f"figure {args.out}_paths.png")
    if args.no_video:
        return
    sides = []
    for n, r in roots.items():
        ep = eps[n]
        front, wrist = decode_episode(r, ep)
        g = data[n][ep]["action"][:, 6] > 0
        sides.append({"label": f"{labels[n]} demo (episode {ep})", "sub": subs[n], "color": COLORS[n],
                      "front": front, "wrist": wrist, "gripper_closed": g, "carry": carry_segment(data[n][ep])})
        print(f"{n}: episode {ep}, {len(front)} frames = {len(front) / CONTROL_HZ:.1f} s")
    write_media(make_frames(sides, instruction), args.out, gif_fps=args.gif_fps)
    print(f"video {args.out}.gif (+ .mp4 if ffmpeg could encode it)")


FONT = _font()

if __name__ == "__main__":
    main()
