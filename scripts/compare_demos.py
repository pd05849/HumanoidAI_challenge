"""Side-by-side comparison of two LIBERO-format LeRobot datasets of the same task: LIBERO's
human-teleoperated demos (run T) and the phone-derived demos (run P).

    python scripts/compare_demos.py --teleop outputs/datasets/teleop_pan \
        --phone ../Humanoid_clips/datasets/phone_pan_P --out docs/img/teleop_vs_phone

Writes
  <out>.gif / <out>.mp4   one demo of each, side by side, front and wrist cameras exactly as the model
                          sees them, played in real time (one dataset frame = one 20 Hz control step,
                          whatever the dataset's fps label says)
  <out>_paths.png         every demo's gripper path: top view and height over time, one column per dataset
  <out>.md                summary table (duration, path length, speed, smoothness, gripper use)

Episodes for the video: --teleop-ep / --phone-ep, by default the one whose length is closest to the
dataset's median. Needs the LeRobot environment (lerobot, matplotlib, opencv); ffmpeg for the mp4
(the gif is written without it if ffmpeg is missing).
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
    """The closed-gripper stretch that moves the gripper furthest sideways: the pan carry (the knob
    turn, whichever order it happens in, moves only a few cm)."""
    p = ep["state"][:, :3]
    segs = closed_segments(ep["action"])
    if not segs:
        return None
    return max(segs, key=lambda se: np.linalg.norm(p[se[1] - 1, :2] - p[se[0], :2]))


def episode_stats(ep):
    s, a = ep["state"], ep["action"]
    p = s[:, :3]                                       # gripper (end-effector) position, m
    t = len(p) / CONTROL_HZ
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    speed = step * CONTROL_HZ
    moving = speed > 0.01                              # > 1 cm/s
    da = np.abs(np.diff(a[:, :6], axis=0)).mean(axis=1)  # change of the arm command per step
    c0, c1 = carry_segment(ep) or (0, len(p))
    return {
        "duration_s": t,
        "path_m": step.sum(),
        "carry_s": (c1 - c0) / CONTROL_HZ,
        "carry_cm": 100 * np.linalg.norm(p[c1 - 1, :2] - p[c0, :2]),
        "carry_lift_cm": 100 * (p[c0:c1, 2].max() - p[c0, 2]),
        "speed_cms": 100 * speed[moving].mean() if moving.any() else 0.0,
        "peak_speed_cms": 100 * speed.max(),
        "action_change": da.mean(),
        "saturated": np.mean(np.any(np.abs(a[:, :6]) > 0.999, axis=1)),
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
    cands = [("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
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
    return lambda size, b=False: ImageFont.load_default()


def _text(img, xy, s, size, color, bold=False):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text(xy, s, font=FONT(size, bold), fill=color)
    img[:] = np.asarray(im)


def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def make_frames(sides, instruction, big=256, small=128, gap=12):
    """sides: list of dicts {label, sub, color, front, wrist, gripper_closed}. Yields RGB frames."""
    n = max(len(s["front"]) for s in sides)
    pw = big + 6 + small                                # panel width
    W = len(sides) * pw + (len(sides) + 1) * gap
    H_TOP, H_HEAD, H_BAR = 40, 50, 40
    H = H_TOP + H_HEAD + big + H_BAR + 8
    surf = _hex(SURFACE)
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
            _text(img, (x0 + 6, y1 + 4), "front camera", 12, (250, 250, 250))
            if i >= T:                                          # finished: dim and say so
                img[y1:y1 + big, x0:x0 + big] = (0.55 * img[y1:y1 + big, x0:x0 + big] + 0.45 * 255).astype(np.uint8)
                _text(img, (x0 + 12, y1 + big // 2 - 12), f"demo ends at {T / CONTROL_HZ:.1f} s", 18, _hex(INK))
            # timeline: grey track, gripper-closed stretches in the dataset colour, playhead
            yb = y1 + big + 12
            g = s["gripper_closed"]
            track = np.full((10, pw, 3), _hex(GRID), np.uint8)
            xs = (np.arange(T) * pw / n).astype(int)
            for t in range(T):
                if g[t]:
                    track[:, xs[t]:max(xs[t] + 1, int((t + 1) * pw / n))] = col
            img[yb:yb + 10, x0:x0 + pw] = track
            xp = x0 + min(int(j * pw / n), pw - 1)
            img[yb - 3:yb + 13, max(xp - 1, x0):xp + 1] = _hex(INK)
            _text(img, (x0, yb + 13), "coloured: gripper closed", 12, _hex(INK2))
        yield img


def write_media(frames, out, gif_fps=10):
    """mp4 at 20 fps (real time) and a gif at gif_fps (every 20/gif_fps-th frame, still real time)."""
    frames = list(frames)
    skip = max(1, round(CONTROL_HZ / gif_fps))
    tmp = Path(tempfile.mkdtemp())
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(str(tmp / f"f{i:04d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(CONTROL_HZ), "-i", str(tmp / "f%04d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", f"{out}.mp4"], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(CONTROL_HZ), "-i", str(tmp / "f%04d.png"),
                            "-vf", f"select='not(mod(n\\,{skip}))',setpts=N/({CONTROL_HZ / skip}*TB),"
                                   "split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];"
                                   "[b][p]paletteuse=dither=none:diff_mode=rectangle",
                            "-r", str(CONTROL_HZ / skip), "-loop", "0", f"{out}.gif"], check=True)
        else:
            from PIL import Image
            ims = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames[::skip]]
            ims[0].save(f"{out}.gif", save_all=True, append_images=ims[1:], duration=int(1000 * skip / CONTROL_HZ),
                        loop=0)
            print("ffmpeg not found: wrote the gif only")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- figure

def landmarks(eps):
    """Median gripper position at the knob turn, the grasp and the release, from demos whose first
    closed-gripper stretch is the knob turn and whose carry comes after it (the phone demos)."""
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
    for name, v in (("stove knob", knob), ("pan handle", grasp), ("burner", rel)):
        if len(v) >= max(3, len(eps) // 2):
            out[name] = np.median(np.array(v), axis=0)
    return out


def make_figure(data, out_png, labels, landmark_source="phone"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    names = list(data)
    fig, ax = plt.subplots(2, len(names), figsize=(5.2 * len(names), 8.4), squeeze=False,
                           gridspec_kw={"height_ratios": [1.15, 1]})
    allp = np.concatenate([e["state"][:, :3] for n in names for e in data[n].values()])
    tmax = max(len(e["state"]) for n in names for e in data[n].values()) / CONTROL_HZ
    marks = landmarks(data[landmark_source]) if landmark_source in data else {}
    for c, n in enumerate(names):
        col = COLORS[n]
        a, b = ax[0, c], ax[1, c]
        for e in data[n].values():
            p = e["state"][:, :3]
            # top view seen from above, robot at the bottom: forward (+x) up, robot's left (+y) to the left
            a.plot(-p[:, 1], p[:, 0], color=col, lw=0.9, alpha=0.35, solid_capstyle="round")
            a.plot(-p[0, 1], p[0, 0], "o", ms=3.5, color=col, mec=SURFACE, mew=0.8, alpha=0.8)
            t = np.arange(len(p)) / CONTROL_HZ
            b.plot(t, p[:, 2], color=col, lw=0.9, alpha=0.35)
        for name, (x, y) in marks.items():          # scene landmarks, same on every panel
            a.plot(-y, x, "o", ms=13, mfc="none", mec=INK, mew=1.1, zorder=5)
            a.annotate(name, (-y, x), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9,
                       color=INK, zorder=6)
        k = len(data[n])
        a.set_title(f"{labels[n]}  ({k} demos)", loc="left", fontsize=11, color=INK, pad=8)
        a.set_xlim(-allp[:, 1].max() - 0.03, -allp[:, 1].min() + 0.03)
        a.set_ylim(allp[:, 0].min() - 0.03, allp[:, 0].max() + 0.03)
        a.set_aspect("equal")
        a.set_xlabel("\u2190 robot's left        sideways (m)        robot's right \u2192")
        if c == 0:
            a.set_ylabel("away from the robot (m) \u2192")
            b.set_ylabel("gripper height (m, simulator frame)")
        b.set_xlim(0, tmax)
        b.set_ylim(allp[:, 2].min() - 0.02, allp[:, 2].max() + 0.02)
        b.set_xlabel("time (s)")
        for x in (a, b):
            x.grid(True, color=GRID, lw=0.6)
            x.set_axisbelow(True)
            for s in x.spines.values():
                s.set_visible(False)
    note = (f"; rings: where the {labels[landmark_source]}s turn the knob, grasp the pan's handle and let go"
            if marks else "")
    fig.suptitle(f"Gripper paths: top view (start = dot{note}) and height over time", x=0.02, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def stats_table(data, labels):
    rows = [("whole demo (s)", "duration_s", "{:.1f}"), ("gripper path length (m)", "path_m", "{:.2f}"),
            ("pan carry: time with the pan in hand (s)", "carry_s", "{:.1f}"),
            ("pan carry: sideways distance (cm)", "carry_cm", "{:.0f}"),
            ("pan carry: highest point above the grasp (cm)", "carry_lift_cm", "{:.0f}"),
            ("gripper speed while moving, mean (cm/s)", "speed_cms", "{:.1f}"),
            ("gripper speed, peak (cm/s)", "peak_speed_cms", "{:.0f}"),
            ("arm command change per step, mean abs. (smaller = smoother)", "action_change", "{:.3f}"),
            ("steps with a saturated arm command (at the +-1 limit)", "saturated", "{:.0%}"),
            ("steps with the gripper closed", "gripper_closed", "{:.0%}")]
    st = {n: [episode_stats(e) for e in data[n].values()] for n in data}
    head = "| median (range) | " + " | ".join(f"{labels[n]} ({len(st[n])})" for n in data) + " |"
    lines = [head, "|---|" + "---|" * len(data)]
    for r in rows:
        name, key, fmt = r[:3]
        mul = r[3] if len(r) > 3 else 1
        cells = []
        for n in data:
            v = np.array([s[key] for s in st[n]]) * mul
            cells.append(f"{fmt.format(np.median(v))} ({fmt.format(v.min())}-{fmt.format(v.max())})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teleop", required=True, help="LeRobot dataset of LIBERO's teleop demos (teleop_pan)")
    ap.add_argument("--phone", required=True, help="LeRobot dataset of phone-derived demos (phone_pan_P)")
    ap.add_argument("--out", default="docs/img/teleop_vs_phone", help="output prefix")
    ap.add_argument("--teleop-ep", type=int, default=None)
    ap.add_argument("--phone-ep", type=int, default=None)
    ap.add_argument("--teleop-label", default="LIBERO teleop demo")
    ap.add_argument("--phone-label", default="phone-derived demo")
    ap.add_argument("--gif-fps", type=float, default=10)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    roots = {"teleop": args.teleop, "phone": args.phone}
    labels = {"teleop": args.teleop_label, "phone": args.phone_label}
    subs = {"teleop": "a person teleoperated the robot",
            "phone": "the robot follows the path of my pan from a phone clip"}
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
    Path(f"{args.out}.md").write_text(table + "\n")
    print(table)
    make_figure(data, f"{args.out}_paths.png", labels)
    print(f"figure {args.out}_paths.png")
    if args.no_video:
        return
    eps = {"teleop": args.teleop_ep, "phone": args.phone_ep}
    sides = []
    for n, r in roots.items():
        ep = eps[n] if eps[n] is not None else median_episode(data[n])
        front, wrist = decode_episode(r, ep)
        g = data[n][ep]["action"][:, 6] > 0
        sides.append({"label": f"{labels[n]} (episode {ep})", "sub": subs[n], "color": COLORS[n],
                      "front": front, "wrist": wrist, "gripper_closed": g})
        print(f"{n}: episode {ep}, {len(front)} frames = {len(front) / CONTROL_HZ:.1f} s")
    write_media(make_frames(sides, instruction), args.out, gif_fps=args.gif_fps)
    print(f"video {args.out}.gif / .mp4")


FONT = _font()

if __name__ == "__main__":
    main()
