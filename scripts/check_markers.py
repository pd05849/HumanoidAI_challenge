"""Check the filming setup before recording the takes.

Record a short test clip (5-10 s) from the iPad's final position: board taped on its burner, pan
on its start spot, then lift the pan and put it on the target burner once. Copy the clip to this
computer and run:

    python scripts/check_markers.py test.mov       # a video
    python scripts/check_markers.py frame.jpg      # or a photo / screenshot

It reports whether the board and the pan marker are found, how large the pan marker appears and
roughly how steeply the camera looks down, and saves annotated frames to check_markers_out/.
Exit code 0 means the setup looks good.
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np

if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
    raise SystemExit("needs OpenCV >= 4.7 with the aruco module: pip install opencv-python")

# The printed markers (assets/markers_letter.pdf, made by scripts/make_markers.py).
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD = cv2.aruco.CharucoBoard((5, 7), 0.030, 0.022, DICT)  # page 1: 5x7 squares of 30 mm
N_CORNERS = 4 * 6                                            # inner corners of a 5x7 board
PAN_ID = 40                                                  # page 2: 80 mm, inside the pan

CHARUCO = cv2.aruco.CharucoDetector(BOARD)
ARUCO = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def rough_camera(w, h):
    """Pinhole guess for a phone/iPad 1x lens (about 60 degrees across the long side).
    Good enough for the look-down angle; the real values come from the calibration clip."""
    f = 0.9 * max(w, h)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def analyse(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    r = {"K": rough_camera(w, h), "corners": 0, "look_down": None, "dist": None,
         "pan_px": None, "pan_edge": False}

    ch_corners, ch_ids, _, _ = CHARUCO.detectBoard(gray)
    if ch_ids is not None and len(ch_ids):
        # same shapes on OpenCV 4.x and 5.x
        ch_corners, ch_ids = ch_corners.reshape(-1, 1, 2), ch_ids.reshape(-1, 1)
        r["corners"] = len(ch_ids)
        r["board_pts"] = (ch_corners, ch_ids)
    if r["corners"] >= 6:
        obj, pts = BOARD.matchImagePoints(ch_corners, ch_ids)
        ok, rvec, tvec = cv2.solvePnP(obj, pts, r["K"], None)
        if ok:
            R = cv2.Rodrigues(rvec)[0]
            centre = R @ np.array([0.075, 0.105, 0.0]) + tvec.ravel()
            # angle between the line of sight and the board's plane: 90 = straight down
            r["look_down"] = math.degrees(math.asin(min(1.0, abs(R[2, 2]))))
            r["dist"] = float(np.linalg.norm(centre))
            r["board_pose"] = (rvec, tvec)

    corners, ids, _ = ARUCO.detectMarkers(gray)
    for c, i in zip(corners, [] if ids is None else ids.ravel()):
        if i == PAN_ID:
            q = c.reshape(4, 2)
            r["pan_px"] = float(np.mean(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)))
            margin = 0.02 * w
            r["pan_edge"] = bool(q.min() < margin or q[:, 0].max() > w - margin
                                 or q[:, 1].max() > h - margin)
            r["pan_quad"] = c
    return r


def annotate(img, r, label):
    vis = img.copy()
    if "board_pts" in r:
        cv2.aruco.drawDetectedCornersCharuco(vis, *r["board_pts"], (0, 255, 0))
    if "board_pose" in r:
        cv2.drawFrameAxes(vis, r["K"], None, *r["board_pose"], 0.06, 3)
    if "pan_quad" in r:
        cv2.aruco.drawDetectedMarkers(vis, [r["pan_quad"]], np.array([[PAN_ID]]), (0, 0, 255))
    lines = [label,
             f"board: {r['corners']}/{N_CORNERS} corners",
             "pan marker: " + (f"{r['pan_px']:.0f} px" if r["pan_px"] else "NOT FOUND")]
    if r["look_down"] is not None:
        lines.append(f"looks down ~{r['look_down']:.0f} deg, board ~{r['dist']:.2f} m (rough)")
    for k, s in enumerate(lines):
        org = (20, 45 + 40 * k)
        cv2.putText(vis, s, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(vis, s, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def frames(path, per_second):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / per_second))
    i = 0
    while cap.grab():
        if i % step == 0:
            ok, img = cap.retrieve()
            if ok:
                yield f"t={i / fps:.1f}s", img
        i += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="test clip (.mov/.mp4) or image (.jpg/.png)")
    ap.add_argument("--per_second", type=float, default=5, help="video frames checked per second")
    ap.add_argument("--out", default="check_markers_out", help="folder for annotated frames")
    args = ap.parse_args()
    path, out = Path(args.path), Path(args.out)
    out.mkdir(exist_ok=True)

    if path.suffix.lower() in IMAGE_EXT:
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"could not read {path}")
        samples = [("image", img)]
    else:
        samples = frames(path, args.per_second)

    results, missed, last = [], 0, None
    for label, img in samples:
        r = analyse(img)
        results.append(r)
        if len(results) == 1:
            cv2.imwrite(str(out / "first.jpg"), annotate(img, r, label))
        if (r["pan_px"] is None or r["corners"] < 6) and missed < 4:
            missed += 1
            cv2.imwrite(str(out / f"missed_{missed}.jpg"), annotate(img, r, label))
        last = (img, r, label)
    if not results:
        raise SystemExit(f"no frames read from {path}")
    if len(results) > 1:
        cv2.imwrite(str(out / "last.jpg"), annotate(*last))

    n = len(results)
    board = [r["corners"] for r in results if r["corners"] >= 6]
    pan = [r["pan_px"] for r in results if r["pan_px"]]
    down = [r["look_down"] for r in results if r["look_down"] is not None]
    dist = [r["dist"] for r in results if r["dist"] is not None]
    problems = []

    print(f"checked {n} frame(s) of {path.name}")
    if board and len(board) >= 0.9 * n and np.median(board) >= 12:
        verdict = "OK"
    else:
        verdict = "CHECK"
        problems.append("Board missing or partly hidden: keep the whole page in view, flat, "
                        "with no glare on it.")
    typical = f", typically {np.median(board):.0f}/{N_CORNERS} corners" if board else ""
    print(f"  board       found in {len(board)}/{n}{typical}   {verdict}")

    verdict = "OK"
    if not pan:
        verdict = "CHECK"
        problems.append("Pan marker (id 40) never found: is it in view, flat and unobstructed?")
    else:
        if np.median(pan) < 40:
            verdict = "CHECK"
            problems.append(f"Pan marker small ({np.median(pan):.0f} px): move the iPad closer.")
        if len(pan) < 0.85 * n:
            verdict = "CHECK"
            problems.append(f"Pan marker lost in {n - len(pan)} of {n} frames (see missed_*.jpg): "
                            "look down more steeply so the pan's rim doesn't hide it, move a bit "
                            "slower, and check for glare on it.")
        if any(r["pan_edge"] for r in results):
            verdict = "CHECK"
            problems.append("Pan marker comes close to the edge of the picture: re-centre the "
                            "iPad or step it back.")
    typical = f", typically {np.median(pan):.0f} px across" if pan else ""
    print(f"  pan marker  found in {len(pan)}/{n}{typical}   {verdict}")

    if down:
        angle = float(np.median(down))
        verdict = "OK" if angle >= 40 else "CHECK"
        if angle < 40:
            problems.append(f"Camera looks down only ~{angle:.0f} deg: raise the iPad and tilt it "
                            "down (aim for 45-60) so the camera sees into the pan.")
        print(f"  camera      looks down ~{angle:.0f} deg, board ~{np.median(dist):.2f} m away   "
              f"{verdict}  (rough, before calibration)")

    print(f"annotated frames: {out}/")
    for p in problems:
        print("  - " + p)
    print("setup looks good" if not problems else "fix the items above and check again")
    raise SystemExit(0 if not problems else 1)


if __name__ == "__main__":
    main()
