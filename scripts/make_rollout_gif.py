"""Side-by-side GIF of evaluation tries, from the videos lerobot-eval saved (the first 10 tries of
every run of train/eval.sh), labelled with the model, the instruction and the outcome.

    python scripts/make_rollout_gif.py base:pan:6 Pall:pan:2 Pall:moka:0 --out docs/img/rollouts.gif

Each panel is NAME:TASK:TRY, i.e. eval_logs/NAME/TASK/videos/**/eval_episode_TRY.mp4, with
TASK = pan (told "frying pan") or moka (told "moka pot") and TRY in 0-9. The outcome comes from
NAME/TASK_episodes.jsonl (in --logs, else in results/). One video frame is one 20 Hz control step;
the GIF plays at real time (every --stride-th step) and holds the last frame of each panel.
Needs PyAV (installed with LeRobot) and Pillow.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CONTROL_HZ = 20
MODELS = {"base": "unchanged smolvla_libero", "T": "T: 50 human demos", "P": "P: 50 phone demos",
          "Pall": "Pall: 153 phone demos"}
TASKS = {"pan": 'told "frying pan"', "moka": 'told "moka pot"'}
INK, INK2, SURFACE = (31, 31, 30), (95, 94, 90), (252, 252, 251)
GOOD, BAD = (0, 99, 0), (178, 34, 34)


def font(size, bold=False):
    try:
        import matplotlib
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)
    except Exception:  # noqa: BLE001 - any missing font falls back to Pillow's own
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def read_frames(path):
    import av
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def find_video(logs, name, task, k):
    hits = sorted((Path(logs) / name / task / "videos").glob(f"**/eval_episode_{k}.mp4"))
    if not hits:
        raise SystemExit(f"no video for {name}:{task}:{k} under {Path(logs) / name / task / 'videos'} "
                         "(lerobot-eval records the first 10 tries only: TRY must be 0-9)")
    return hits[0]


def outcome(logs, name, task, k):
    for root in (Path(logs), Path("results")):
        p = root / name / f"{task}_episodes.jsonl"
        if p.exists():
            recs = [json.loads(line) for line in open(p)]
            if k < len(recs):
                r = recs[k]
                first = r.get("first_object_picked_up") or ""
                what = "frying pan" if "frypan" in first else ("moka pot" if "moka" in first else "nothing")
                return bool(r["success"]), f"lifted first: {what}"
    return None, ""


HEAD, FOOT = 46, 46


def panel(frame, size, title, sub, t, result):
    img = Image.new("RGB", (size, HEAD + size + FOOT), SURFACE)
    img.paste(Image.fromarray(frame).resize((size, size), Image.BILINEAR), (0, HEAD))
    d = ImageDraw.Draw(img)
    d.text((6, 4), title, font=font(14, True), fill=INK)
    d.text((6, 24), sub, font=font(13), fill=INK2)
    d.text((6, HEAD + size + 5), f"{t:4.1f} s", font=font(13), fill=INK2)
    if result is not None:
        ok, detail = result
        d.text((size - 6, HEAD + size + 5), "success" if ok else "failed", font=font(13, True),
               fill=GOOD if ok else BAD, anchor="ra")
        d.text((6, HEAD + size + 25), detail, font=font(12), fill=INK2)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panels", nargs="+", help="NAME:TASK:TRY, e.g. base:pan:6")
    ap.add_argument("--logs", default="eval_logs")
    ap.add_argument("--out", default="docs/img/rollouts.gif")
    ap.add_argument("--size", type=int, default=256, help="panel width in pixels")
    ap.add_argument("--stride", type=int, default=2, help="keep every n-th control step (2 = 10 frames/s)")
    ap.add_argument("--hold", type=float, default=2.0, help="seconds to hold the last frame")
    ap.add_argument("--colors", type=int, default=128)
    args = ap.parse_args()

    clips = []
    for spec in args.panels:
        name, task, k = spec.split(":")
        k = int(k)
        frames = read_frames(find_video(args.logs, name, task, k))
        ok, detail = outcome(args.logs, name, task, k)
        clips.append(dict(frames=frames, title=MODELS.get(name, name), sub=TASKS.get(task, task),
                          result=(ok, detail) if ok is not None else None))
        print(f"{spec}: {len(frames)} steps ({len(frames) / CONTROL_HZ:.1f} s), "
              f"{'success' if ok else 'failed' if ok is not None else 'outcome unknown'} {detail}")

    n = max(len(c["frames"]) for c in clips)
    steps = list(range(0, n, args.stride)) + [n - 1] * max(1, round(args.hold * CONTROL_HZ / args.stride))
    gap = np.full((HEAD + args.size + FOOT, 6, 3), SURFACE, np.uint8)
    out = []
    for i in steps:
        tiles = []
        for c in clips:
            j = min(i, len(c["frames"]) - 1)
            done = j == len(c["frames"]) - 1
            tiles += [panel(c["frames"][j], args.size, c["title"], c["sub"], j / CONTROL_HZ,
                            c["result"] if done else None), gap]
        out.append(Image.fromarray(np.concatenate(tiles[:-1], axis=1)))

    picks = [np.asarray(f) for f in out[:: max(1, len(out) // 8)] + [out[-1]]]
    w = picks[0].shape[1]  # big blocks of the text colours, so the palette keeps them exact
    picks += [np.full((60, w, 3), c, np.uint8) for c in (INK, INK2, SURFACE, GOOD, BAD)]
    sample = Image.fromarray(np.concatenate(picks, axis=0))
    palette = sample.quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT)
    gif = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in out]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    gif[0].save(args.out, save_all=True, append_images=gif[1:], duration=int(1000 * args.stride / CONTROL_HZ),
                loop=0, optimize=True, disposal=1)
    mb = Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out}: {len(gif)} frames, {mb:.1f} MB" + ("  (large: try --stride 3 or --size 224)" if mb > 8 else ""))


if __name__ == "__main__":
    main()
