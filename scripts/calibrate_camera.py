"""Camera calibration from clips of the ChArUco board (page 1 of assets/markers_letter.pdf).

    python scripts/calibrate_camera.py calib_00.mov calib_01.mov ... --out data/calibration/ipad.json

Hold the board in front of the fixed camera and move/tilt it through the whole picture. If the
second copy of the board (taped on the burner) is also in view, that is fine: both copies carry
the same marker ids, so in every frame only the large markers (the held board, close to the
camera) are kept, one per id, before the board corners are interpolated.

Lens model: pinhole + radial distortion k1, k2 (no tangential terms, no k3). With this model
independent clips at 30 and 60 fps agreed to 0.2 % in focal length and 3 px in principal point;
the richer models fitted the same data only marginally better and disagreed between clips.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD = cv2.aruco.CharucoBoard((5, 7), 0.030, 0.022, DICT)
BOARD_IDS = 16                                   # board markers are ids 0..16
FLAGS = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3


def held_board_points(gray, marker_det, charuco_det, min_corners):
    """Board corners of the held (largest) board copy, or None."""
    corners, ids, _ = marker_det.detectMarkers(gray)
    if ids is None:
        return None
    keep = [(q.reshape(4, 2), int(i)) for q, i in zip(corners, ids.ravel()) if i <= BOARD_IDS]
    if not keep:
        return None
    side = np.array([np.linalg.norm(q - np.roll(q, 1, 0), axis=1).mean() for q, _ in keep])
    best = {}
    for (q, i), s in zip(keep, side):
        if s >= 0.6 * side.max() and (i not in best or s > best[i][1]):
            best[i] = (q, s)
    if len(best) < 4:
        return None
    mc = tuple(best[i][0].reshape(1, 4, 2).astype(np.float32) for i in best)
    mi = np.array(list(best), np.int32).reshape(-1, 1)
    cc, ci, _, _ = charuco_det.detectBoard(gray, markerCorners=mc, markerIds=mi)
    if ci is None or len(ci) < min_corners:
        return None
    return BOARD.matchImagePoints(cc.reshape(-1, 1, 2), ci.reshape(-1, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out", required=True, help="output JSON")
    ap.add_argument("--every", type=float, default=0.1, help="seconds between sampled frames")
    ap.add_argument("--min-corners", type=int, default=12)
    args = ap.parse_args()

    marker_det = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
    charuco_det = cv2.aruco.CharucoDetector(BOARD)
    obj, img, size, sampled = [], [], None, 0
    for clip in args.clips:
        cap = cv2.VideoCapture(clip)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(fps * args.every))
        k = 0
        while cap.grab():
            if k % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    sampled += 1
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    size = gray.shape[::-1]
                    pts = held_board_points(gray, marker_det, charuco_det, args.min_corners)
                    if pts is not None:
                        obj.append(pts[0]), img.append(pts[1])
            k += 1
        cap.release()
    if len(obj) < 20:
        raise SystemExit(f"only {len(obj)} usable frames: move the board through more of the picture")

    w, h = size
    K0 = np.array([[0.95 * w, 0, w / 2], [0, 0.95 * w, h / 2], [0, 0, 1]])
    rms, K, dist, rvecs, _, std_int, _, per_view = cv2.calibrateCameraExtended(
        obj, img, size, K0, None, flags=FLAGS)
    std_int, per_view = std_int.ravel(), per_view.ravel()
    pts = np.vstack([i.reshape(-1, 2) for i in img])
    cover, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[4, 3], range=[[0, w], [0, h]])
    tilts = [float(np.degrees(np.arccos(abs(cv2.Rodrigues(r)[0][2, 2])))) for r in rvecs]

    out = dict(
        image_size=[int(w), int(h)],
        K=K.tolist(), dist=dist.ravel()[:5].tolist(),
        model="pinhole + radial k1, k2 (no tangential, no k3); OpenCV conventions",
        rms_px=float(rms), worst_view_px=float(per_view.max()),
        std=dict(fx=float(std_int[0]), fy=float(std_int[1]), cx=float(std_int[2]), cy=float(std_int[3])),
        frames_used=len(obj), frames_sampled=sampled,
        coverage_corners_4x3=cover.T.astype(int).tolist(),
        board_tilt_deg=dict(min=min(tilts), median=float(np.median(tilts)), max=max(tilts)),
        clips=[Path(c).name for c in args.clips],
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{len(obj)}/{sampled} frames, rms {rms:.3f} px (worst view {per_view.max():.2f} px)")
    print(f"fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}  "
          f"k1 {dist.ravel()[0]:.4f}  k2 {dist.ravel()[1]:.4f}")
    print("corner coverage (4 columns x 3 rows):")
    print(cover.T.astype(int))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
