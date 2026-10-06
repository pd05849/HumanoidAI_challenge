"""Draw the tracked pan path on top of the original clip, so the tracking can be checked by eye.

    python scripts/render_tracking.py clip.mov tracks/clip/path.json --out tracks/clip/tracking.mp4

What is drawn (all re-projected from the tracked 3-D pose, not copied from the detector):
- the 80 mm marker square at the tracked pose, coloured by where the measurement came from
  (green = marker detected, yellow = template-tracked through a hard frame, red = filled gap),
  and the pan's own axes (x red, y green, z blue = "up" of the pan);
- the path of the marker centre so far (blue), its shadow on the counter plane (grey) and
  vertical "stalks" between them every 0.25 s, which show the height;
- the world axes on the board (x right, y toward the wall, z up, 10 cm);
- a live panel: phase, height above the start, distance travelled, rotation, and a height-vs-time
  graph with a cursor.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

R_WB = np.diag([1.0, -1.0, -1.0])         # world <- OpenCV board frame (see track_pan.py)
T_WB = np.array([0.0, 0.21, 0.0])
HALF = 0.040                               # half side of the pan marker (m)
SRC_COL = {"aruco": (80, 200, 60), "aruco_permissive": (80, 200, 60),
           "ecc_tracked": (0, 210, 255), "interpolated": (40, 40, 230), "rejected_outlier": (40, 40, 230)}


def euler_to_R(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def text(img, s, org, scale=0.55, col=(255, 255, 255), th=1):
    # plain text on a dark band (an outline drawn with a thicker stroke is offset in OpenCV 5)
    (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = org
    roi = img[max(0, y - h - 5):y + base + 3, max(0, x - 5):x + w + 5]
    roi[:] = (0.35 * roi).astype(roi.dtype)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("path_json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--fps", type=float, default=30.0, help="output frame rate (frames are skipped)")
    args = ap.parse_args()

    doc = json.loads(Path(args.path_json).read_text())
    meta, rows, smooth = doc["metadata"], doc["path"], doc["path_smoothed"]
    cam = meta["camera"]
    K = np.array(cam["K"], float)
    dist = None if isinstance(cam["distortion"], str) else np.array(cam["distortion"], float)
    rvec, tvec = np.array(cam["board_rvec"]), np.array(cam["board_tvec"])
    ph, fps = meta["phases"], meta["fps"]

    def proj(pw):
        pb = (R_WB.T @ (np.atleast_2d(pw) - T_WB).T).T
        return cv2.projectPoints(pb, rvec, tvec, K, dist)[0].reshape(-1, 2)

    P = np.array([[r["x"], r["y"], r["z"]] for r in smooth])
    A = np.array([[r["roll"], r["pitch"], r["yaw"]] for r in smooth])
    start = np.array(meta["metrics"]["start_m"])
    z_rel = 100 * (P[:, 2] - start[2])
    yaw0 = A[0, 2]
    trail = proj(P)
    shadow = P.copy()
    shadow[:, 2] = start[2]
    trail_sh = proj(shadow)
    origin = proj([0, 0, 0])[0]
    axes_w = [proj(v)[0] for v in ([0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1])]

    def phase(t):
        def g(k):
            return ph.get(k) if ph.get(k) is not None else float("inf")
        if t < g("first_disturbance_s"):
            return "rest (before)"
        if t < g("lift_off_s"):
            return "grasp"
        if t < g("peak_height_s"):
            return "lift + carry"
        if t < g("touchdown_s"):
            return "carry + lower"
        if t < g("settled_s"):
            return "release"
        return "rest (after)"

    cap = cv2.VideoCapture(args.video)
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = args.width / w0
    W, H = args.width, int(round(h0 * s))
    step = max(1, int(round(fps / args.fps)))
    tmp = Path(tempfile.mkdtemp()) / "raw.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (W, H))
    # height graph geometry (bottom-left)
    gx, gy, gw, gh = 12, H - 132, 300, 110
    t_all = np.arange(len(P)) / fps
    zmax = max(5.0, float(np.nanmax(z_rel)) + 2)

    def gpt(t, z):
        return (int(gx + gw * t / t_all[-1]), int(gy + gh - gh * (z + 1) / (zmax + 1)))

    graph_pts = np.array([gpt(t, z) for t, z in zip(t_all, z_rel)], np.int32)
    k = 0
    while True:
        ok, img = cap.read()
        if not ok or k >= len(P):
            break
        if k % step:
            k += 1
            continue
        t = k / fps
        # world axes on the board
        for e, col in zip(axes_w, ((0, 0, 255), (0, 200, 0), (255, 0, 0))):
            cv2.arrowedLine(img, tuple(np.round(origin).astype(int)), tuple(np.round(e).astype(int)), col, 3,
                            cv2.LINE_AA, tipLength=0.2)
        # trail, shadow and height stalks so far
        if k > 1:
            cv2.polylines(img, [np.round(trail_sh[:k + 1]).astype(np.int32).reshape(-1, 1, 2)], False,
                          (170, 170, 170), 3, cv2.LINE_AA)
            for j in range(0, k + 1, max(1, int(round(0.25 * fps)))):
                cv2.line(img, tuple(np.round(trail[j]).astype(int)), tuple(np.round(trail_sh[j]).astype(int)),
                         (200, 200, 200), 1, cv2.LINE_AA)
            cv2.polylines(img, [np.round(trail[:k + 1]).astype(np.int32).reshape(-1, 1, 2)], False,
                          (235, 140, 40), 4, cv2.LINE_AA)
        # marker square and pan axes at the tracked pose
        Rw = euler_to_R(*A[k])
        sq = np.array([[-HALF, HALF, 0], [HALF, HALF, 0], [HALF, -HALF, 0], [-HALF, -HALF, 0]])
        quad = proj(P[k] + (Rw @ sq.T).T)
        src = rows[k]["source"]
        cv2.polylines(img, [np.round(quad).astype(np.int32).reshape(-1, 1, 2)], True, SRC_COL.get(src, (255, 255, 255)),
                      4, cv2.LINE_AA)
        c = proj(P[k])[0]
        for v, col in ((Rw[:, 0], (0, 0, 255)), (Rw[:, 1], (0, 200, 0)), (Rw[:, 2], (255, 0, 0))):
            e = proj(P[k] + 0.07 * v)[0]
            cv2.arrowedLine(img, tuple(np.round(c).astype(int)), tuple(np.round(e).astype(int)), col, 3,
                            cv2.LINE_AA, tipLength=0.2)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        # panel
        d_xy = 100 * np.linalg.norm(P[k, :2] - start[:2])
        dyaw = np.degrees(A[k, 2] - yaw0)
        name = Path(args.video).stem
        text(img, f"{name}   t = {t:5.2f} s   phase: {phase(t)}", (12, 26), 0.65, (255, 255, 255), 2)
        text(img, f"height above start {z_rel[k]:5.1f} cm   moved {d_xy:5.1f} cm   turned {dyaw:+5.1f} deg",
             (12, 52), 0.6)
        text(img, f"tracking: {src.replace('_', ' ')}", (12, 76), 0.55, SRC_COL.get(src, (255, 255, 255)))
        ov = img.copy()
        cv2.rectangle(ov, (gx - 6, gy - 22), (gx + gw + 6, gy + gh + 6), (0, 0, 0), -1)
        img = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
        text(img, "height above start (cm)", (gx, gy - 6), 0.45)
        cv2.polylines(img, [graph_pts.reshape(-1, 1, 2)], False, (235, 140, 40), 2, cv2.LINE_AA)
        cx_, cy_ = gpt(t, z_rel[k])
        cv2.line(img, (cx_, gy), (cx_, gy + gh), (255, 255, 255), 1)
        cv2.circle(img, (cx_, cy_), 4, (255, 255, 255), -1, cv2.LINE_AA)
        vw.write(img)
        k += 1
    vw.release()
    cap.release()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out], check=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
