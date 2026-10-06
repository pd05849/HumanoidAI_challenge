"""Pan tracker: phone clip -> pan path in a board-fixed world frame.

    python scripts/track_pan.py clip.mov --calib data/calibration/ipad_1080p.json --out out_dir

With --calib the camera intrinsics and lens distortion come from scripts/calibrate_camera.py.
Without it the focal length is estimated from the static board (single view; less accurate).

Pipeline
  1. Every frame: ChArUco board (fixed reference lying on the cooktop) and the pan marker
     (DICT_4X4_50 id 40, 80 mm black square, flat on the inside bottom of the pan).
  2. Camera-static check from the spread of the board corners over the clip; if static, ONE board
     pose is used for the whole clip (solved on the per-corner median image points).
  3. Focal length from the static board (single-view calibration: principal point fixed at the
     image centre, square pixels, no distortion) + cross-checks: bootstrap, per-frame spread,
     principal-point / k1 sensitivity, "pan level at rest" consistency, QuickTime metadata.
  4. Pan marker corners: ArUco (SUBPIX), a permissive second pass for missed frames, then
     ECC template alignment (homography) of the whole marker pattern. The ECC step fixes the
     corner that the pan's rim pulls off by ~10 px in many frames, and tracks the marker through
     frames where ArUco finds nothing.
  5. Pose per frame with solvePnPGeneric(IPPE_SQUARE); planar ambiguity resolved by taking the
     solution whose normal is closest to the board normal (the pan stays roughly level).
  6. World-frame path, outlier rejection, gap filling, smoothing, rest windows, lift-off /
     touchdown, speed, and a systematic-error budget (focal +-10 %, principal point +-30 px,
     k1 +-0.05, marker print scale +-1 %).
Outputs in --out: path.json, path.png, overlay.jpg, detections.npz (cache).
"""
import argparse
import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
SQUARES = (5, 7)
SQ, MK = 0.030, 0.022
BOARD = cv2.aruco.CharucoBoard(SQUARES, SQ, MK, DICT)
N_CORNERS = (SQUARES[0] - 1) * (SQUARES[1] - 1)
BOARD_W, BOARD_H = SQUARES[0] * SQ, SQUARES[1] * SQ           # 0.15 x 0.21 m
OBJ_CORNERS = BOARD.getChessboardCorners().astype(np.float64)  # (24,3) in the board frame
PAN_ID, PAN_SIDE = 40, 0.080
PAN_OBJ = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], np.float64) * PAN_SIDE / 2

# World frame from the board. OpenCV board frame: origin at the pattern's outer corner that is the
# top-left of the printed page, x along the 5-square side (page right), y along the 7-square side
# (down the page), z = x cross y = INTO the board. The page lies with its top toward the wall, so
# x_b points right along the counter, y_b toward the person, z_b down (checked from the data: the
# camera centre must come out at negative z_b, i.e. positive world z).
R_WB = np.diag([1.0, -1.0, -1.0])
T_WB = np.array([0.0, BOARD_H, 0.0])
WORLD_FRAME = (
    "Right-handed, metres. Origin = front-left outer corner of the 5x7 ChArUco pattern lying on the "
    "front-left burner (the pattern corner nearest the person and furthest left; OpenCV board point "
    "(0, 0.21, 0)). x = along the board's 5-square edge, pointing RIGHT along the counter (toward the "
    "knobs and the pan's start spot); y = along the 7-square edge, pointing toward the WALL (away from "
    "the person); z = UP, normal to the board/glass plane, z = 0 at the top of the board paper. "
    "x_w = x_board, y_w = 0.21 m - y_board, z_w = -z_board. Pan position = centre of the 80 mm "
    "marker on the inside bottom of the pan. Orientation = ZYX Euler angles (yaw about world z, then "
    "pitch about y, then roll about x) of the marker frame (x, y in the marker plane, z out of its "
    "printed face = up when the pan is level). The marker's in-plane rotation relative to the "
    "handle is unknown, so yaw is the heading of the marker's x edge: use it as a change, not as "
    "the handle direction.")

# ECC template of the marker (black square + part of the white quiet zone)
TPL_MOD, TPL_BORDER = 12, 0.5            # px per marker module, white border in modules


def _template():
    m = cv2.aruco.generateImageMarker(DICT, PAN_ID, 6 * TPL_MOD)
    b = int(round(TPL_BORDER * TPL_MOD))
    t = np.full((6 * TPL_MOD + 2 * b,) * 2, 255, np.uint8)
    t[b:b + 6 * TPL_MOD, b:b + 6 * TPL_MOD] = m
    s = 6 * TPL_MOD
    c = np.array([[b, b], [b + s, b], [b + s, b + s], [b, b + s]], np.float64) - 0.5
    return t.astype(np.float32), c


TPL, TPL_CORNERS = _template()


# ----------------------------------------------------------------------------------------------
# detection
def board_detectors():
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 3, 33, 4
    p.minMarkerPerimeterRate = 0.01
    p.polygonalApproxAccuracyRate = 0.05
    p.perspectiveRemovePixelPerCell = 8
    p.perspectiveRemoveIgnoredMarginPerCell = 0.2
    cp = cv2.aruco.CharucoParameters()
    cp.tryRefineMarkers = True        # recovers the small board markers the first pass misses
    plain = cv2.aruco.CharucoDetector(BOARD, cp, cv2.aruco.DetectorParameters())
    return [cv2.aruco.CharucoDetector(BOARD, cp, p), plain]


def detect_board(dets, gray):
    best = np.full((N_CORNERS, 2), np.nan)
    for d in dets:
        cc, ci, _, _ = d.detectBoard(gray)
        if ci is not None and len(ci) > (~np.isnan(best[:, 0])).sum():
            best = np.full((N_CORNERS, 2), np.nan)
            best[np.asarray(ci).ravel()] = np.asarray(cc).reshape(-1, 2)
        if (~np.isnan(best[:, 0])).sum() == N_CORNERS:
            break
    return best


def pan_detectors():
    strict = cv2.aruco.DetectorParameters()
    strict.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    loose = cv2.aruco.DetectorParameters()
    loose.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    loose.adaptiveThreshWinSizeMin, loose.adaptiveThreshWinSizeMax = 3, 53
    loose.adaptiveThreshWinSizeStep = 4
    loose.polygonalApproxAccuracyRate = 0.08
    loose.perspectiveRemoveIgnoredMarginPerCell = 0.3
    loose.maxErroneousBitsInBorderRate = 0.6
    loose.errorCorrectionRate = 1.0
    loose.minDistanceToBorder = 1
    return cv2.aruco.ArucoDetector(DICT, strict), cv2.aruco.ArucoDetector(DICT, loose)


def find_pan(det, gray):
    corners, ids, _ = det.detectMarkers(gray)
    best = None
    for c, i in zip(corners, [] if ids is None else ids.ravel()):
        if i == PAN_ID:
            q = c.reshape(4, 2).astype(np.float64)
            area = cv2.contourArea(q.astype(np.float32))
            if best is None or area > best[1]:
                best = (q, area)
    return None if best is None else best[0]


def ecc_refine(gray, q0):
    """Align the marker template to the image (homography, ECC). Returns corners, correlation."""
    h, w = gray.shape
    x0, y0 = np.floor(q0.min(0) - 40).astype(int)
    x1, y1 = np.ceil(q0.max(0) + 40).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None, 0.0
    roi = np.ascontiguousarray(gray[y0:y1, x0:x1]).astype(np.float32)
    H0 = cv2.getPerspectiveTransform(TPL_CORNERS.astype(np.float32),
                                     (q0 - [x0, y0]).astype(np.float32)).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        cc, H = cv2.findTransformECC(TPL, roi, H0, cv2.MOTION_HOMOGRAPHY, crit, None, 3)
    except cv2.error:
        return None, 0.0
    c = cv2.perspectiveTransform(TPL_CORNERS.reshape(-1, 1, 2).astype(np.float32), H).reshape(4, 2)
    return c.astype(np.float64) + [x0, y0], float(cc)


def detect(video):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    bdets = board_detectors()
    strict, loose = pan_detectors()
    board, aruco, src, ecc, ecc_cc = [], [], [], [], []
    prev = []                                   # refined corners of the last two frames
    while True:
        ok, img = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        board.append(detect_board(bdets, g))
        q, s = find_pan(strict, g), 1
        if q is None:
            q, s = find_pan(loose, g), 2        # rim-occluded / blurred: permissive second pass
        aruco.append(np.full((4, 2), np.nan) if q is None else q)
        init = q
        if q is None and prev and prev[-1][0] == len(board) - 2:   # track from the last frame
            init = prev[-1][1]
            if len(prev) > 1 and prev[-2][0] == len(board) - 3:
                init = 2 * prev[-1][1] - prev[-2][1]               # constant velocity
            s = 3
        c, cc = (None, 0.0) if init is None else ecc_refine(g, init)
        if c is None or cc < (0.80 if s == 3 else 0.5):
            if s == 3:
                s = 0
            c = q                                                    # fall back to ArUco corners
            cc = 0.0 if c is None else cc
        elif s in (1, 2):
            pass
        src.append(s if (c is not None) else 0)
        ecc.append(np.full((4, 2), np.nan) if c is None else c)
        ecc_cc.append(cc)
        if c is not None:
            prev = (prev + [(len(board) - 1, c)])[-2:]
    cap.release()
    if not board:
        raise SystemExit(f"no frames read from {video}")
    h, w = g.shape
    # src: 0 none, 1 ArUco strict, 2 ArUco permissive pass, 3 ECC-tracked (no ArUco detection)
    return dict(fps=float(fps), size=(w, h), board=np.array(board), aruco=np.array(aruco),
                pan=np.array(ecc), pan_src=np.array(src), ecc_cc=np.array(ecc_cc))


# ----------------------------------------------------------------------------------------------
# camera
def K_of(f, w, h, dx=0.0, dy=0.0):
    return np.array([[f, 0, w / 2.0 + dx], [0, f, h / 2.0 + dy], [0, 0, 1.0]])


def quicktime_focal(video, w, h):
    """com.apple.quicktime.camera.focal_length.35mm_equivalent -> px under three conventions."""
    try:
        data = Path(video).read_bytes()
    except OSError:
        return None

    def boxes(s, e):
        i = s
        while i + 8 <= e:
            size, typ = struct.unpack(">I4s", data[i:i + 8])
            hdr = 8
            if size == 1:
                size, hdr = struct.unpack(">Q", data[i + 8:i + 16])[0], 16
            elif size == 0:
                size = e - i
            if size < hdr:
                return
            yield typ, i + hdr, i + size
            i += size

    vals = {}

    def metas(s, e):          # every meta box under moov (movie level and per track)
        for typ, a, b in boxes(s, e):
            if typ == b"meta":
                yield a, b
            elif typ in (b"moov", b"trak", b"udta", b"mdia", b"minf"):
                yield from metas(a, b)

    try:
        for a2, b2 in metas(0, len(data)):
            if True:
                keys = []
                for t3, a3, b3 in boxes(a2, b2):
                    if t3 == b"keys":
                        n, j = struct.unpack(">I", data[a3 + 4:a3 + 8])[0], a3 + 8
                        for _ in range(n):
                            ks = struct.unpack(">I", data[j:j + 4])[0]
                            keys.append(data[j + 8:j + ks].decode("latin1"))
                            j += ks
                    if t3 == b"ilst":
                        for t4, a4, b4 in boxes(a3, b3):
                            idx = struct.unpack(">I", t4)[0]
                            for t5, a5, b5 in boxes(a4, b4):
                                if t5 == b"data" and 0 < idx <= len(keys):
                                    tp = struct.unpack(">I", data[a5:a5 + 4])[0]
                                    v = data[a5 + 8:b5]
                                    if tp == 23:
                                        vals[keys[idx - 1]] = struct.unpack(">f", v)[0]
                                    elif tp in (21, 22):
                                        vals[keys[idx - 1]] = int.from_bytes(v, "big", signed=tp == 21)
                                    elif tp == 1:
                                        vals[keys[idx - 1]] = v.decode("utf8", "replace")
    except Exception:
        return None
    f35 = vals.get("com.apple.quicktime.camera.focal_length.35mm_equivalent")
    try:
        f35 = float(f35)
    except (TypeError, ValueError):
        return None
    half_diag_35 = math.hypot(36, 24) / 2
    return dict(f35mm=float(f35), lens=vals.get("com.apple.quicktime.camera.lens_model"),
                model=vals.get("com.apple.quicktime.model"),
                px_if_diag_of_16x9_frame=float(math.hypot(w, h) / 2 * f35 / half_diag_35),
                px_if_width_36mm=float(w * f35 / 36.0),
                px_if_diag_of_4x3_sensor=float((w / 2) * f35 / (0.8 * half_diag_35)),
                note="Apple does not document which frame size the video value refers to, and a "
                     "stabilisation crop would make the true f larger, so this only says f is "
                     "of order 1500-1800 px")


def pose_from_board(img_pts, ids, K, dist=None):
    obj = OBJ_CORNERS[ids]
    ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE)
    rvec, tvec = cv2.solvePnPRefineLM(obj, img_pts, K, dist, rvec, tvec)
    proj = cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - img_pts) ** 2, 1))))
    return cv2.Rodrigues(rvec)[0], tvec.ravel(), rms


CALIB_FLAGS = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
               | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)


def calib_f(obj_list, img_list, w, h, f0, K0=None, dist0=None):
    K0 = K_of(f0, w, h) if K0 is None else K0
    out = cv2.calibrateCameraExtended([o.astype(np.float32) for o in obj_list],
                                      [i.astype(np.float32) for i in img_list], (w, h), K0,
                                      np.zeros(5) if dist0 is None else dist0, flags=CALIB_FLAGS)
    rms, K, std_int = out[0], out[1], out[5].ravel()
    return float(K[0, 0]), float(max(std_int[0], std_int[1])), float(rms)


def homography_focal(img_pts, ids, w, h):
    """Closed form from H = K[r1 r2 t] with known principal point and square pixels."""
    H, _ = cv2.findHomography(OBJ_CORNERS[ids][:, :2], img_pts - np.array([w / 2, h / 2]))
    h1, h2 = H[:, 0], H[:, 1]
    f2a = -(h1[0] * h2[0] + h1[1] * h2[1]) / (h1[2] * h2[2])
    f2b = -(h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / (h1[2] ** 2 - h2[2] ** 2)
    return [float(math.sqrt(v)) if v > 0 else None for v in (f2a, f2b)]


def estimate_focal(det, w, h):
    B, n = det["board"], len(det["board"])
    med, ids = det["board_ref"], det["board_ref_ids"]
    f_guess = 0.9 * max(w, h)
    est = {"guess_0.9_width": f_guess}
    f1, s1, rms1 = calib_f([OBJ_CORNERS[ids]], [med], w, h, f_guess)
    est["board_single_view"] = dict(f=f1, std_formal=s1, rms_px=rms1,
                                    how="calibrateCamera on the per-corner median image points, "
                                        "principal point = image centre, fx = fy, no distortion")
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(200):
        sel = rng.choice(len(ids), len(ids), replace=True)
        if len(set(sel)) < 10:
            continue
        try:
            boot.append(calib_f([OBJ_CORNERS[ids[sel]]], [med[sel]], w, h, f1)[0])
        except cv2.error:
            pass
    est["bootstrap_corners"] = dict(std=float(np.std(boot)), p05=float(np.percentile(boot, 5)),
                                    p95=float(np.percentile(boot, 95)))
    nb = (~np.isnan(B[:, :, 0])).sum(1)
    per = []
    for k in np.where(nb == N_CORNERS)[0]:
        try:
            per.append(calib_f([OBJ_CORNERS], [B[k]], w, h, f1)[0])
        except cv2.error:
            pass
    if per:
        est["per_frame"] = dict(n=len(per), median=float(np.median(per)), std=float(np.std(per)))
    fa, fb = homography_focal(med, ids, w, h)
    est["homography_closed_form"] = dict(orthogonality=fa, equal_norm=fb)
    sys_ = {}
    for dx, dy in ((30, 0), (-30, 0), (0, 30), (0, -30)):
        sys_[f"pp_{dx:+d}_{dy:+d}px"] = calib_f([OBJ_CORNERS[ids]], [med], w, h, f1,
                                                 K0=K_of(f1, w, h, dx, dy))[0]
    for k1 in (-0.05, 0.05):
        sys_[f"k1_{k1:+.2f}"] = calib_f([OBJ_CORNERS[ids]], [med], w, h, f1,
                                         dist0=np.array([k1, 0, 0, 0, 0.0]))[0]
    est["board_single_view_if"] = sys_
    pv = [k for k in range(n) if det["pan_src"][k] in (1, 2)]
    if len(pv) > 10:
        try:
            fp, sp, rp = calib_f([PAN_OBJ] * len(pv), [det["pan"][k] for k in pv], w, h, f1)
            est["pan_marker_views_only"] = dict(
                f=fp, std_formal=sp, rms_px=rp, views=len(pv),
                note="weak: an 80 mm square has little perspective, and sub-pixel edge biases "
                     "(blur, rolling shutter while moving) pull it; not used")
        except cv2.error:
            pass
    est["quicktime_metadata"] = quicktime_focal(det["video"], w, h)
    return est


# ----------------------------------------------------------------------------------------------
# pose
def rot_to_euler_zyx(R):
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    return roll, pitch, yaw


def pan_poses(pan, K, R_cb, t_cb, dist=None, side=PAN_SIDE):
    n = len(pan)
    obj = PAN_OBJ * side / PAN_SIDE
    up_cam = -R_cb[:, 2]                       # board normal toward the camera (= world up)
    R_w_c = R_WB @ R_cb.T
    out = dict(pos=np.full((n, 3), np.nan), R=np.full((n, 3, 3), np.nan), err=np.full(n, np.nan),
               err_other=np.full(n, np.nan), tilt_other=np.full(n, np.nan),
               dist=np.full(n, np.nan), px=np.full(n, np.nan))
    for k in range(n):
        q = pan[k]
        if np.isnan(q).any():
            continue
        if dist is not None:
            q = cv2.undistortPoints(q.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
        _, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, q, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        sols = []
        for rv, tv, e in zip(rvecs, tvecs, np.ravel(errs)):
            R = cv2.Rodrigues(rv)[0]
            sols.append((float(R[:, 2] @ up_cam), R, tv.ravel(), float(e)))
        sols.sort(key=lambda s: -s[0])          # normal closest to the board normal first
        _, R_cm, t_cm, e = sols[0]
        out["err"][k] = e
        if len(sols) > 1:
            out["err_other"][k] = sols[1][3]
            out["tilt_other"][k] = math.degrees(math.acos(np.clip(sols[1][0], -1, 1)))
        out["R"][k] = R_w_c @ R_cm
        out["pos"][k] = R_WB @ (R_cb.T @ (t_cm - t_cb)) + T_WB
        out["dist"][k] = float(np.linalg.norm(t_cm))
        out["px"][k] = float(np.mean(np.linalg.norm(pan[k] - np.roll(pan[k], 1, 0), axis=1)))
    return out


# ----------------------------------------------------------------------------------------------
# time-series helpers
def runs(mask):
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        if not m and s is not None:
            out.append((s, i - 1))
            s = None
    if s is not None:
        out.append((s, len(mask) - 1))
    return out


def fmt_runs(rr):
    return [f"{a}" if a == b else f"{a}-{b}" for a, b in rr]


def hermite_fill(t, v, valid):
    """Fill invalid samples with a cubic Hermite spline through the valid ones."""
    out = v.copy()
    tv, vv = t[valid], v[valid]
    miss = ~valid
    if len(tv) < 2 or not miss.any():
        return out
    m = np.gradient(vv, tv)
    idx = np.clip(np.searchsorted(tv, t[miss]), 1, len(tv) - 1)
    t0, t1, y0, y1, m0, m1 = tv[idx - 1], tv[idx], vv[idx - 1], vv[idx], m[idx - 1], m[idx]
    hs = t1 - t0
    s = np.clip((t[miss] - t0) / hs, 0, 1)
    out[miss] = ((2 * s ** 3 - 3 * s ** 2 + 1) * y0 + (s ** 3 - 2 * s ** 2 + s) * hs * m0
                 + (-2 * s ** 3 + 3 * s ** 2) * y1 + (s ** 3 - s ** 2) * hs * m1)
    return out


def savgol(v, win, order=2):
    n, half = len(v), win // 2
    out = np.empty(n)
    for i in range(n):
        a = min(max(0, i - half), max(0, n - win))
        xs = np.arange(a, min(n, a + win)) - i
        A = np.vander(xs, order + 1, increasing=True)
        out[i] = (np.linalg.pinv(A) @ v[a:a + len(xs)])[0]
    return out


def loo_residual(t, v, valid, half=4, order=2):
    """Residual of each valid sample against a local polynomial fitted to its valid neighbours."""
    res = np.full(len(v), np.nan)
    for i in np.where(valid)[0]:
        nb = [j for j in range(max(0, i - half), min(len(v), i + half + 1)) if valid[j] and j != i]
        if len(nb) < order + 3:
            continue
        x = t[nb] - t[i]
        c = np.polyfit(x, v[nb], order)
        res[i] = v[i] - c[-1]
    return res


# ----------------------------------------------------------------------------------------------
def analyse_path(det, f, dx=0.0, dy=0.0, dist=None, side=PAN_SIDE, base_radius=0.09, K=None):
    w, h = det["size"]
    fps = det["fps"]
    K = K_of(f, w, h, dx, dy) if K is None else np.asarray(K, float)
    f = float(K[0, 0])
    ref, ref_ids = det["board_ref"], det["board_ref_ids"]
    if dist is not None:
        ref = cv2.undistortPoints(ref.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    R_cb, t_cb, board_rms = pose_from_board(ref, ref_ids, K)
    cam_w = R_WB @ (-R_cb.T @ t_cb) + T_WB
    P = pan_poses(det["pan"], K, R_cb, t_cb, dist, side)
    n = len(det["pan"])
    t = np.arange(n) / fps
    measured = ~np.isnan(P["pos"][:, 0])

    euler = np.full((n, 3), np.nan)
    tilt = np.full(n, np.nan)
    for k in np.where(measured)[0]:
        euler[k] = rot_to_euler_zyx(P["R"][k])
        tilt[k] = math.degrees(math.acos(np.clip(P["R"][k][2, 2], -1, 1)))
    ang = euler.copy()
    ang[measured, 2] = np.unwrap(euler[measured, 2])

    # outliers: residual to a local quadratic through the neighbours (position, tilt), large
    # reprojection error, or poor ECC correlation
    bad = np.zeros(n, bool)
    for v, floor in [(P["pos"][:, 0], 0.006), (P["pos"][:, 1], 0.006), (P["pos"][:, 2], 0.006),
                     (tilt, 2.5)]:
        r = loo_residual(t, v, measured)
        s = 1.4826 * np.nanmedian(np.abs(r))
        bad |= np.abs(np.nan_to_num(r)) > max(6 * s, floor)
    bad |= measured & (P["err"] > 1.5)
    # physical plausibility: inside the kitchen workspace (m, world frame) ...
    pos = P["pos"]
    inside = ((pos[:, 0] > -0.5) & (pos[:, 0] < 1.5) & (pos[:, 1] > -0.5) & (pos[:, 1] < 1.0)
              & (pos[:, 2] > -0.15) & (pos[:, 2] < 0.6))
    bad |= measured & ~inside
    # ... and no jumps: within 8 cm of the median of the other samples in +-0.25 s (a smooth hand
    # carry moves well under that from the window centre; a wrong quad or a flipped pose does not)
    half = max(2, int(round(0.25 * fps)))
    ok_idx = np.where(measured & ~bad)[0]
    for k in ok_idx:
        nb = ok_idx[(np.abs(ok_idx - k) <= half) & (ok_idx != k)]
        if len(nb) >= 3 and np.linalg.norm(pos[k] - np.median(pos[nb], 0)) > 0.08:
            bad[k] = True
    good = measured & ~bad

    pos_f = np.column_stack([hermite_fill(t, P["pos"][:, c], good) for c in range(3)])
    ang_f = np.column_stack([hermite_fill(t, ang[:, c], good) for c in range(3)])
    tilt_f = hermite_fill(t, tilt, good)
    win = max(5, int(round(0.25 * fps)) | 1)
    pos_s = np.column_stack([savgol(pos_f[:, c], win) for c in range(3)])
    ang_s = np.column_stack([savgol(ang_f[:, c], win) for c in range(3)])
    vel = np.gradient(pos_s, t, axis=0)
    speed, speed_h = np.linalg.norm(vel, axis=1), np.linalg.norm(vel[:, :2], axis=1)

    # rest windows: deviation from the first / last 0.5 s of good samples
    n0 = max(5, int(0.5 * fps))
    gi = np.where(good)[0]
    first, last = gi[:n0], gi[-n0:]
    p0, p1 = np.median(P["pos"][first], 0), np.median(P["pos"][last], 0)
    a0, a1 = np.median(ang[first], 0), np.median(ang[last], 0)
    # thresholds: 6 sigma of the noise in that 0.5 s, at least 1.5 mm / 0.5 deg
    thr = []
    for sel in (first, last):
        sp = float(np.sqrt(P["pos"][sel].var(0).sum()))
        sa = float(np.max(ang[sel].std(0)))
        thr.append((max(0.0015, 6 * sp), max(math.radians(0.5), 6 * sa)))

    def moving(p, a, th):
        return ((np.linalg.norm(pos_s - p, axis=1) > th[0])
                | (np.max(np.abs(ang_s - a), axis=1) > th[1]))

    m0, m1 = moving(p0, a0, thr[0]), moving(p1, a1, thr[1])
    onset = int(np.argmax(m0)) if m0.any() else n
    settle = int(n - np.argmax(m1[::-1])) if m1.any() else 0     # first frame of the final rest
    margin = int(round(0.1 * fps))
    rest0 = (0, max(0, onset - margin - 1))
    rest1 = (min(n - 1, settle + margin), n - 1)
    idx = np.arange(n)
    r0, r1 = good & (idx <= rest0[1]), good & (idx >= rest1[0])
    start, end = np.median(P["pos"][r0], 0), np.median(P["pos"][r1], 0)

    # lift-off / touchdown on the smoothed height
    z = pos_s[:, 2]
    i_peak = int(np.argmax(z))
    lifted = np.where((idx < i_peak) & (z - start[2] <= 0.010))[0]
    i_lift10 = int(lifted[-1] + 1) if len(lifted) else None           # last crossing of +1 cm
    pre = np.where((idx < (i_lift10 or 0)) & (z - start[2] <= 0.003))[0]
    i_lift = int(pre[-1]) if len(pre) else None                         # left the +3 mm band
    down = np.where((idx > i_peak) & (z - end[2] <= 0.003))[0]
    i_touch = int(down[0]) if len(down) else None                      # back within +3 mm
    down10 = np.where((idx > i_peak) & (z - end[2] <= 0.010))[0]
    i_touch10 = int(down10[0]) if len(down10) else None
    # the pan pivots on an edge of its base while tilted: estimate the lowest point of the base
    # (disc of radius base_radius in the marker plane) and when it leaves / reaches the surface
    tilt_s = savgol(tilt_f, win)
    z_low = z - base_radius * np.sin(np.radians(np.clip(tilt_s, 0, None)))
    off = np.where((idx < i_peak) & (z_low - start[2] <= 0.002))[0]
    i_base_off = int(off[-1]) if len(off) else None                     # whole base clear (est.)
    on = np.where((idx > i_peak) & (z_low - end[2] <= 0.002))[0]
    i_base_on = int(on[0]) if len(on) else None                         # first edge contact (est.)

    def jitter(mask):
        p, a = P["pos"][mask], np.degrees(ang[mask])
        return dict(frames=int(mask.sum()),
                    pos_std_mm=[round(float(v), 3) for v in 1000 * p.std(0)],
                    pos_std_3d_mm=float(1000 * np.sqrt(p.var(0).sum())),
                    roll_std_deg=float(a[:, 0].std()), pitch_std_deg=float(a[:, 1].std()),
                    yaw_std_deg=float(a[:, 2].std()), tilt_std_deg=float(np.std(tilt[mask])),
                    tilt_mean_deg=float(np.mean(tilt[mask])))

    tg = np.where(good, tilt, -1)
    return dict(
        focal_px=float(f), K=K, dist=dist, R_cb=R_cb, t_cb=t_cb, board_rms_px=board_rms,
        camera_world_m=[float(v) for v in cam_w],
        look_down_deg=float(math.degrees(math.asin(abs(R_cb[2, 2])))),
        board_dist_m=float(np.linalg.norm(R_cb @ np.array([BOARD_W / 2, BOARD_H / 2, 0]) + t_cb)),
        t=t, P=P, measured=measured, good=good, bad=bad, ang=ang, tilt=tilt,
        pos_f=pos_f, ang_f=ang_f, tilt_f=tilt_f, pos_s=pos_s, ang_s=ang_s, speed=speed,
        speed_h=speed_h, rest0=rest0, rest1=rest1, onset=onset, settle=settle,
        start=start, end=end,
        start_end_dist_h=float(np.linalg.norm((end - start)[:2])),
        start_end_dist_3d=float(np.linalg.norm(end - start)),
        lift_max=float(np.max(P["pos"][good, 2]) - start[2]),
        lift_max_smooth=float(z.max() - start[2]),
        i_peak=i_peak, i_lift=i_lift, i_lift10=i_lift10, i_touch=i_touch, i_touch10=i_touch10,
        i_base_off=i_base_off, i_base_on=i_base_on, z_low=z_low, rest_thresholds=thr,
        jitter0=jitter(r0), jitter1=jitter(r1),
        max_tilt=float(tg.max()), i_max_tilt=int(tg.argmax()),
        yaw_change=float(math.degrees(np.median(ang[r1, 2]) - np.median(ang[r0, 2]))),
        yaw_min_max=[float(math.degrees(np.min(ang[good, 2]) - np.median(ang[r0, 2]))),
                     float(math.degrees(np.max(ang[good, 2]) - np.median(ang[r0, 2])))],
        peak_speed=float(speed.max()), i_peak_speed=int(speed.argmax()),
        peak_speed_h=float(speed_h.max()),
    )


# ----------------------------------------------------------------------------------------------
# outputs
def save_plot(res, det, gaps_mask, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, INK, MUTED, GRID, REST = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
    t, n, fps = res["t"], len(res["t"]), det["fps"]
    good, src = res["good"], det["pan_src"]
    pos, pos_s = res["P"]["pos"], res["pos_s"]
    tracked = good & (src == 3)
    win = max(5, int(round(0.25 * fps)) | 1)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 9.5,
                         "axes.titlelocation": "left"})
    fig = plt.figure(figsize=(13, 15), facecolor="white")
    gs = fig.add_gridspec(4, 2, height_ratios=[1.8, 1.0, 1, 1], hspace=0.42, wspace=0.2)
    ax_top = fig.add_subplot(gs[0, :])
    ax_side = fig.add_subplot(gs[1, :])
    ax_z = fig.add_subplot(gs[2, 0])
    ax_yaw = fig.add_subplot(gs[2, 1], sharex=ax_z)
    ax_tilt = fig.add_subplot(gs[3, 0], sharex=ax_z)
    ax_v = fig.add_subplot(gs[3, 1], sharex=ax_z)

    # top view (x-y) and side view (x-z), same x scale
    ax_top.fill(100 * np.array([0, BOARD_W, BOARD_W, 0]), 100 * np.array([0, 0, BOARD_H, BOARD_H]),
                color=REST, zorder=1)
    ax_top.text(100 * BOARD_W / 2, 100 * BOARD_H / 2, "ChArUco\nboard", ha="center", va="center",
                color=MUTED, fontsize=8)
    ax_top.plot(100 * pos_s[:, 0], 100 * pos_s[:, 1], color=BLUE, lw=2, zorder=2,
                label="smoothed path (gaps filled)")
    ax_top.plot(100 * pos[good, 0], 100 * pos[good, 1], "o", ms=2.5, color=BLUE, alpha=0.45,
                zorder=3, label="measured frames")
    if gaps_mask.any():
        ax_top.plot(100 * res["pos_f"][gaps_mask, 0], 100 * res["pos_f"][gaps_mask, 1], "x",
                    color=ORANGE, ms=7, mew=1.5, zorder=4, label="no measurement (interpolated)")
    if tracked.any():
        ax_top.plot(100 * pos[tracked, 0], 100 * pos[tracked, 1], "s", mfc="none", color=ORANGE,
                    ms=7, mew=1.5, zorder=4, label="ECC-tracked (ArUco found nothing)")
    for p, lab, ha in ((res["start"], "start", "center"), (res["end"], "end", "center")):
        ax_top.plot(100 * p[0], 100 * p[1], "o", ms=10, mfc="white", mec=INK, mew=1.5, zorder=5)
        ax_top.annotate(f"{lab} ({100 * p[0]:.1f}, {100 * p[1]:.1f}) cm", (100 * p[0], 100 * p[1]),
                        textcoords="offset points", xytext=(0, 11), ha=ha, fontsize=8.5, color=INK)
    allx = 100 * np.concatenate([pos_s[:, 0], [0, BOARD_W]])
    xlim = (np.floor(allx.min()) - 3, np.ceil(allx.max()) + 3)
    ally = 100 * np.concatenate([pos_s[:, 1], [0, BOARD_H]])
    ax_top.set_xlim(*xlim)
    ax_top.set_ylim(np.floor(ally.min()) - 2, np.ceil(ally.max()) + 4)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_ylabel("y (cm), toward the wall")
    ax_top.set_title("Top view, world frame (origin = front-left corner of the board, x right along "
                     "the counter, y toward the wall)", color=INK)
    ax_top.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    ax_side.plot(100 * pos_s[:, 0], 100 * pos_s[:, 2], color=BLUE, lw=2)
    ax_side.plot(100 * pos[good, 0], 100 * pos[good, 2], "o", ms=2.5, color=BLUE, alpha=0.45)
    ax_side.axhline(0, color=MUTED, lw=0.8)
    ax_side.set_xlim(*xlim)
    ax_side.set_ylim(min(-1.5, 100 * pos_s[:, 2].min() - 1), 100 * pos_s[:, 2].max() + 2)
    ax_side.set_aspect("equal", adjustable="box")
    ax_side.set_xlabel("x (cm), right along the counter")
    ax_side.set_ylabel("z (cm), up")
    ax_side.set_title("Side view (x-z), same scale; z = 0 is the top of the board", color=INK)

    def shade(ax):
        for a, b in runs(gaps_mask):
            ax.axvspan(t[a] - 0.5 / fps, t[b] + 0.5 / fps, color=ORANGE, alpha=0.25, lw=0)
        for k in np.where(tracked)[0]:
            ax.axvline(t[k], color=ORANGE, lw=1.2, alpha=0.8)
        for a, b in (res["rest0"], res["rest1"]):
            ax.axvspan(t[a], t[b], color=REST, alpha=0.5, lw=0)
        for i in (res["i_lift"], res["i_touch"]):
            if i is not None:
                ax.axvline(t[i], color=MUTED, lw=0.9, ls="--")

    def series(ax, smooth, raw, ylabel, ttl):
        ax.plot(t, smooth, color=BLUE, lw=2)
        ax.plot(t[good], raw[good], "o", ms=2, color=BLUE, alpha=0.45)
        shade(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(ttl, color=INK)

    z0 = res["start"][2]
    series(ax_z, 100 * (pos_s[:, 2] - z0), 100 * (pos[:, 2] - z0), "cm",
           "Height above the start rest (marker centre)")
    y0 = np.median(res["ang"][good & (np.arange(n) <= res["rest0"][1]), 2])
    series(ax_yaw, np.degrees(res["ang_s"][:, 2] - y0), np.degrees(res["ang"][:, 2] - y0), "deg",
           "Yaw change about world z")
    series(ax_tilt, savgol(res["tilt_f"], win), res["tilt"], "deg",
           "Tilt (marker normal vs world up)")
    ax_v.plot(t, 100 * res["speed"], color=BLUE, lw=2)
    shade(ax_v)
    ax_v.set_ylabel("cm/s")
    ax_v.set_title("Speed of the smoothed path (3D)", color=INK)
    for ax in (ax_tilt, ax_v):
        ax.set_xlabel("time (s)")
    fig.text(0.02, 0.005, "Time panels: grey = rest windows, dashed = lift-off / touchdown (marker "
             "centre leaves / returns within 3 mm), orange band = no measurement, orange line = "
             "ECC-tracked frame (ArUco found nothing).", fontsize=8, color=MUTED)
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_overlay(video, res, det, path):
    cap = cv2.VideoCapture(str(video))
    n = len(det["pan"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return
    K, R_cb, t_cb = res["K"], res["R_cb"], res["t_cb"]
    rvec = cv2.Rodrigues(R_cb)[0]

    def proj(pw):        # world -> image
        pb = (R_WB.T @ (np.atleast_2d(pw) - T_WB).T).T
        return cv2.projectPoints(pb, rvec, t_cb, K, res.get("dist"))[0].reshape(-1, 2)

    # world axes at the origin (10 cm)
    o = proj([0, 0, 0])[0]
    for ax_, col in (([0.1, 0, 0], (0, 0, 255)), ([0, 0.1, 0], (0, 200, 0)), ([0, 0, 0.1], (255, 0, 0))):
        e = proj(ax_)[0]
        cv2.arrowedLine(img, tuple(np.round(o).astype(int)), tuple(np.round(e).astype(int)), col, 3,
                        cv2.LINE_AA, tipLength=0.15)
    # path and its shadow on the board plane
    p = res["pos_s"]
    shadow = p.copy()
    shadow[:, 2] = 0
    for pts, col, th in ((proj(shadow), (160, 160, 160), 2), (proj(p), (214, 120, 42), 3)):
        cv2.polylines(img, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, col, th, cv2.LINE_AA)
    for k in range(0, n, 10):
        q = proj(p[k])[0]
        cv2.circle(img, tuple(np.round(q).astype(int)), 4, (52, 104, 235), -1, cv2.LINE_AA)
    for s_, lab in (((20, 40), "world axes: x red (right), y green (to wall), z blue (up), 10 cm"),
                    ((20, 80), "marker-centre path (blue), dots every 10 frames (orange); "
                               "grey = its shadow on z = 0")):
        cv2.putText(img, lab, s_, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, lab, s_, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", default="track_out")
    ap.add_argument("--calib", default=None, help="camera calibration JSON from calibrate_camera.py")
    ap.add_argument("--focal", type=float, default=None,
                    help="focal length in px (default: estimated from the static board)")
    ap.add_argument("--redetect", action="store_true", help="ignore the cached detections")
    ap.add_argument("--detect-only", action="store_true", help="only run (and cache) the marker detection")
    ap.add_argument("--no-stabilise", action="store_true",
                    help="do not subtract the per-frame image shift of the board from the pan marker")
    ap.add_argument("--base-radius", type=float, default=0.09,
                    help="radius of the pan's flat base in m, only for the edge-contact estimates")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video = Path(args.video)

    cache = out / "detections.npz"
    stamp = f"{video.resolve()}|{video.stat().st_size}|{video.stat().st_mtime}|v3"
    keys = ("board", "aruco", "pan", "pan_src", "ecc_cc")
    if cache.exists() and not args.redetect and str(np.load(cache)["stamp"]) == stamp:
        zc = np.load(cache)
        det = dict(fps=float(zc["fps"]), size=tuple(int(v) for v in zc["size"]),
                   **{k: zc[k] for k in keys})
        print(f"using cached detections {cache}")
    else:
        print(f"detecting markers in {video} ...")
        det = detect(video)
        np.savez_compressed(cache, stamp=stamp, fps=det["fps"], size=det["size"],
                            **{k: det[k] for k in keys})
    if args.detect_only:
        return
    det["video"] = video
    w, h = det["size"]
    n, fps = len(det["pan"]), det["fps"]
    print(f"{n} frames, {w}x{h}, {fps:.3f} fps")

    # ---- board: detection + static check
    B = det["board"]
    nb = (~np.isnan(B[:, :, 0])).sum(1)
    board_ok = nb >= 6
    if not board_ok.any():
        raise SystemExit("board not found")
    med = np.nanmedian(B[board_ok], axis=0)
    ref_ids = np.where(~np.isnan(med[:, 0]))[0]
    dev = np.linalg.norm(B[board_ok] - med, axis=2)
    per_frame_dev = np.nanmean(dev, axis=1)
    drift = np.nanmean(B[board_ok] - med, axis=1)                   # mean image shift per frame
    static = bool(np.percentile(per_frame_dev, 95) < 1.0 and np.abs(drift).max() < 2.0)
    det["board_ref"], det["board_ref_ids"] = med[ref_ids], ref_ids
    # camera micro-motion (stabilisation / tripod): the mean shift of the board corners per frame.
    # The pan marker moves with it in the image (correlation > 0.9 at rest), so subtract it.
    shift = np.full((n, 2), np.nan)
    okk = nb >= 12
    shift[okk] = np.nanmean(B[okk] - med, axis=1)
    for c in range(2):
        shift[:, c] = np.interp(np.arange(n), np.where(okk)[0], shift[okk, c])
    det["pan_raw"] = det["pan"].copy()
    if not args.no_stabilise:
        det["pan"] = det["pan"] - shift[:, None, :]

    # ---- camera: calibration file, or focal estimated from the static board
    K_cal = dist_cal = None
    if args.calib:
        cal = json.loads(Path(args.calib).read_text())
        if list(cal["image_size"]) != [w, h]:
            raise SystemExit(f"calibration is for {cal['image_size']}, video is {w}x{h}")
        K_cal, dist_cal = np.array(cal["K"], float), np.array(cal["dist"], float)
        focal = dict(calibration=cal)
        f_use, f_src = float(K_cal[0, 0]), f"calibration {Path(args.calib).name}"
    else:
        focal = estimate_focal(det, w, h)
        f_board = focal["board_single_view"]["f"]
        if args.focal:
            f_use, f_src = args.focal, "--focal"
        elif math.isfinite(f_board) and focal["bootstrap_corners"]["std"] / f_board < 0.03:
            f_use, f_src = f_board, "board single-view estimate"
        else:
            f_use, f_src = 0.9 * max(w, h), "0.9 x image width guess (board estimate too uncertain)"
        # rest-level consistency: pan should be level (parallel to the board) at both rests
        scan = []
        for sc in np.linspace(0.85, 1.15, 13):
            r = analyse_path(det, f_use * sc)
            scan.append(dict(f=r["focal_px"], board_rms_px=r["board_rms_px"],
                             start_tilt_deg=r["jitter0"]["tilt_mean_deg"],
                             end_tilt_deg=r["jitter1"]["tilt_mean_deg"],
                             start_z_mm=1000 * float(r["start"][2]), end_z_mm=1000 * float(r["end"][2]),
                             dist_h_cm=100 * r["start_end_dist_h"], lift_cm=100 * r["lift_max"]))
        focal["scan_vs_f"] = scan
        fs = np.array([d["f"] for d in scan])
        tl = np.array([d["start_tilt_deg"] + d["end_tilt_deg"] for d in scan])
        c = np.polyfit(fs, tl, 2)
        focal["rest_level_best_f"] = float(-c[1] / (2 * c[0])) if c[0] > 0 else None

    # ---- main analysis
    res = analyse_path(det, f_use, base_radius=args.base_radius, K=K_cal, dist=dist_cal)
    other = dict(det, pan=det["pan_raw"] if not args.no_stabilise else det["pan_raw"] - shift[:, None, :])
    res_alt = analyse_path(other, f_use, base_radius=args.base_radius, K=K_cal, dist=dist_cal)
    jitter_alt = dict(start=res_alt["jitter0"], end=res_alt["jitter1"],
                      what="without" if not args.no_stabilise else "with")
    # sensitivity of the main numbers to the camera model and the marker print scale
    K_base = res["K"]
    d_base = np.zeros(5) if dist_cal is None else dist_cal

    def k_mod(sf=1.0, dx=0.0, dy=0.0):
        K2 = K_base.copy()
        K2[0, 0] *= sf
        K2[1, 1] *= sf
        K2[0, 2] += dx
        K2[1, 2] += dy
        return K2

    variants = {"focal_-3%": dict(K=k_mod(0.97)), "focal_+3%": dict(K=k_mod(1.03)),
                "pp_x_-20px": dict(K=k_mod(dx=-20)), "pp_x_+20px": dict(K=k_mod(dx=20)),
                "pp_y_-20px": dict(K=k_mod(dy=-20)), "pp_y_+20px": dict(K=k_mod(dy=20)),
                "k1_-0.02": dict(dist=d_base + [-0.02, 0, 0, 0, 0]),
                "k1_+0.02": dict(dist=d_base + [0.02, 0, 0, 0, 0]),
                "pan_marker_-1%": dict(side=0.99 * PAN_SIDE), "pan_marker_+1%": dict(side=1.01 * PAN_SIDE)}
    base = dict(start_end_dist_h_cm=100 * res["start_end_dist_h"], lift_max_cm=100 * res["lift_max"],
                start_z_mm=1000 * float(res["start"][2]), end_z_mm=1000 * float(res["end"][2]),
                start_tilt_deg=res["jitter0"]["tilt_mean_deg"], end_tilt_deg=res["jitter1"]["tilt_mean_deg"],
                yaw_change_deg=res["yaw_change"], peak_speed_cm_s=100 * res["peak_speed"])
    sens = {"nominal": dict(focal_px=float(K_base[0, 0]), **base)}
    for name, kw in variants.items():
        kw = {**dict(K=K_base, dist=dist_cal), **kw}
        r = analyse_path(det, f_use, **kw)
        sens[name] = dict(focal_px=r["focal_px"], start_end_dist_h_cm=100 * r["start_end_dist_h"],
                          lift_max_cm=100 * r["lift_max"], start_z_mm=1000 * float(r["start"][2]),
                          end_z_mm=1000 * float(r["end"][2]),
                          start_tilt_deg=r["jitter0"]["tilt_mean_deg"],
                          end_tilt_deg=r["jitter1"]["tilt_mean_deg"], yaw_change_deg=r["yaw_change"],
                          peak_speed_cm_s=100 * r["peak_speed"], board_rms_px=r["board_rms_px"])

    # ---- detection bookkeeping
    t = res["t"]
    src = det["pan_src"]
    aruco_found = src in (1, 2) if np.isscalar(src) else np.isin(src, (1, 2))
    measured = res["measured"]
    no_meas = ~measured | res["bad"]
    tt = lambda i: None if i is None else round(float(t[i]), 3)
    P = res["P"]
    chosen_lower_err = np.mean((P["err"] <= P["err_other"])[measured])

    # ---- JSON
    path_rows = []
    for k in range(n):
        g = bool(res["good"][k])
        row = dict(t=round(float(t[k]), 4), frame=k, detected=bool(aruco_found[k]),
                   measured=g, source=["none", "aruco", "aruco_permissive", "ecc_tracked"][int(src[k])]
                   if g else ("rejected_outlier" if measured[k] else "interpolated"))
        p = res["P"]["pos"][k] if g else res["pos_f"][k]
        a = res["ang"][k] if g else res["ang_f"][k]
        row.update(x=round(float(p[0]), 5), y=round(float(p[1]), 5), z=round(float(p[2]), 5),
                   roll=round(float(a[0]), 5), pitch=round(float(a[1]), 5), yaw=round(float(a[2]), 5),
                   tilt_deg=round(float(res["tilt"][k] if g else res["tilt_f"][k]), 3),
                   ecc_corr=round(float(det["ecc_cc"][k]), 4))
        path_rows.append(row)
    smooth_rows = [dict(t=round(float(t[k]), 4), frame=k,
                        x=round(float(res["pos_s"][k, 0]), 5), y=round(float(res["pos_s"][k, 1]), 5),
                        z=round(float(res["pos_s"][k, 2]), 5), roll=round(float(res["ang_s"][k, 0]), 5),
                        pitch=round(float(res["ang_s"][k, 1]), 5), yaw=round(float(res["ang_s"][k, 2]), 5))
                   for k in range(n)]
    ecc_low = [int(k) for k in np.where(measured & (det["ecc_cc"] < 0.92))[0]]
    meta = dict(
        video=str(video), frames=n, fps=fps, image_size=[w, h],
        camera=dict(focal_px=f_use, focal_source=f_src, K=res["K"].tolist(),
                    principal_point=[float(res["K"][0, 2]), float(res["K"][1, 2])],
                    distortion=("none assumed" if dist_cal is None else [float(v) for v in dist_cal]),
                    static=static,
                    board_rvec=[float(v) for v in cv2.Rodrigues(res["R_cb"])[0].ravel()],
                    board_tvec=[float(v) for v in res["t_cb"]],
                    board_pose_from="one pose for all frames, solved on the per-corner median image points"
                    if static else "WARNING camera not static; median board pose used anyway",
                    camera_centre_world_m=res["camera_world_m"], look_down_deg=res["look_down_deg"],
                    board_distance_m=res["board_dist_m"], board_reprojection_rms_px=res["board_rms_px"]),
        world_frame=WORLD_FRAME,
        board=dict(corners_per_frame_median=float(np.median(nb)), corners_min=int(nb.min()),
                   frames_found=int(board_ok.sum()),
                   corner_deviation_from_median_px=dict(median=float(np.median(per_frame_dev)),
                                                        p95=float(np.percentile(per_frame_dev, 95)),
                                                        max=float(per_frame_dev.max())),
                   mean_image_shift_px_range=[[float(v) for v in drift.min(0)], [float(v) for v in drift.max(0)]],
                   image_shift_compensation=not args.no_stabilise),
        focal_estimates=focal,
        detection=dict(
            aruco_strict=int(np.sum(src == 1)), aruco_permissive_pass=int(np.sum(src == 2)),
            ecc_tracked_without_aruco=int(np.sum(src == 3)), not_measured=int(np.sum(~measured)),
            rejected_outliers=[int(k) for k in np.where(res["bad"])[0]],
            aruco_strict_missed_ranges=fmt_runs(runs(src != 1)),
            aruco_missed_ranges=fmt_runs(runs(~aruco_found)),
            no_measurement_ranges=fmt_runs(runs(no_meas)),
            ecc_correlation_below_0p92=ecc_low,
            ambiguity=dict(min_tilt_of_rejected_ippe_solution_deg=float(np.nanmin(P["tilt_other"])),
                           frac_frames_chosen_solution_has_lower_reproj_error=float(chosen_lower_err),
                           reproj_err_chosen_px_median=float(np.nanmedian(P["err"])),
                           reproj_err_other_px_median=float(np.nanmedian(P["err_other"])))),
        phases=dict(
            rest_start_frames=list(res["rest0"]), rest_start_s=[tt(res["rest0"][0]), tt(res["rest0"][1])],
            rest_end_frames=list(res["rest1"]), rest_end_s=[tt(res["rest1"][0]), tt(res["rest1"][1])],
            rest_thresholds_mm_deg=[[round(1000 * a, 2), round(math.degrees(b), 2)]
                                    for a, b in res["rest_thresholds"]],
            first_disturbance_s=tt(res["onset"]),
            lift_off_s=tt(res["i_lift"]), base_clear_est_s=tt(res["i_base_off"]),
            above_1cm_s=tt(res["i_lift10"]), peak_height_s=tt(res["i_peak"]),
            back_below_1cm_s=tt(res["i_touch10"]), base_first_contact_est_s=tt(res["i_base_on"]),
            touchdown_s=tt(res["i_touch"]), settled_s=tt(res["settle"]),
            carry_duration_s=(None if res["i_lift"] is None or res["i_touch"] is None
                              else round(float(t[res["i_touch"]] - t[res["i_lift"]]), 3)),
            base_radius_assumed_m=args.base_radius,
            definitions="first_disturbance = smoothed pose leaves the start pose by more than the rest "
                        "threshold (6 sigma of the first 0.5 s, >= 1.5 mm / 0.5 deg), i.e. the hand "
                        "first moves the pan; the start rest window ends 0.1 s before it. lift_off = last "
                        "frame before the climb with the marker centre within 3 mm of its start height; "
                        "touchdown = first frame after the peak with the centre within 3 mm of its end "
                        "height; settled = first frame after which the pose stays within the end rest "
                        "threshold (the end rest window starts 0.1 s later). base_clear_est / "
                        "base_first_contact_est use the lowest point of a base disc of base_radius in "
                        "the marker plane (the pan pivots on one edge while tilted), threshold 2 mm."),
        metrics=dict(
            start_m=[float(v) for v in res["start"]], end_m=[float(v) for v in res["end"]],
            start_end_distance_horizontal_m=res["start_end_dist_h"],
            start_end_distance_3d_m=res["start_end_dist_3d"],
            lift_max_above_start_m=res["lift_max"], lift_max_above_start_smoothed_m=res["lift_max_smooth"],
            max_tilt_deg=res["max_tilt"], max_tilt_at_s=tt(res["i_max_tilt"]),
            yaw_change_deg=res["yaw_change"], yaw_excursion_deg=res["yaw_min_max"],
            peak_speed_m_s=res["peak_speed"], peak_speed_at_s=tt(res["i_peak_speed"]),
            peak_horizontal_speed_m_s=res["peak_speed_h"],
            rest_jitter_start=res["jitter0"], rest_jitter_end=res["jitter1"],
            rest_jitter_other_stabilisation_setting=jitter_alt),
        sensitivity=sens,
    )
    doc = dict(metadata=meta, path=path_rows, path_smoothed=smooth_rows)
    (out / "path.json").write_text(json.dumps(doc, indent=1))

    title = (f"{video.name}: pan marker path  |  f = {f_use:.0f} px ({f_src})  |  start->end "
             f"{100 * res['start_end_dist_h']:.1f} cm, lift {100 * res['lift_max']:.1f} cm, "
             f"peak {100 * res['peak_speed']:.0f} cm/s")
    save_plot(res, det, no_meas, out / "path.png", title)
    save_overlay(video, res, det, out / "overlay.jpg")

    # ---- console summary
    print(f"board: {np.median(nb):.0f}/{N_CORNERS} corners (min {nb.min()}), static={static} "
          f"(corner dev median {np.median(per_frame_dev):.2f} px, p95 {np.percentile(per_frame_dev, 95):.2f}, "
          f"max {per_frame_dev.max():.2f}); camera at {np.round(res['camera_world_m'], 3)} m, looks down "
          f"{res['look_down_deg']:.1f} deg, board {res['board_dist_m']:.2f} m away")
    if K_cal is None:
        fb = focal["board_single_view"]
        print(f"focal: board {fb['f']:.0f} (formal +-{fb['std_formal']:.0f}, bootstrap +-"
              f"{focal['bootstrap_corners']['std']:.0f}), rest-level best {focal['rest_level_best_f']}; "
              f"using {f_use:.0f}")
    else:
        print(f"camera: {f_src} (fx {K_cal[0, 0]:.0f}, fy {K_cal[1, 1]:.0f}, cx {K_cal[0, 2]:.0f}, "
              f"cy {K_cal[1, 2]:.0f}, k1 {dist_cal[0]:.3f}, k2 {dist_cal[1]:.3f})")
    print(f"pan: ArUco strict {np.sum(src == 1)}, permissive {np.sum(src == 2)}, ECC-tracked "
          f"{np.sum(src == 3)}, none {np.sum(~measured)}; outliers {fmt_runs(runs(res['bad']))}; "
          f"ArUco missed {fmt_runs(runs(~aruco_found))}")
    print(f"start {np.round(100 * res['start'], 2)} cm, end {np.round(100 * res['end'], 2)} cm, "
          f"dist_h {100 * res['start_end_dist_h']:.2f} cm, lift {100 * res['lift_max']:.2f} cm, "
          f"max tilt {res['max_tilt']:.1f} deg @ {tt(res['i_max_tilt'])}s, yaw change "
          f"{res['yaw_change']:.1f} deg, peak speed {100 * res['peak_speed']:.1f} cm/s @ "
          f"{tt(res['i_peak_speed'])}s")
    print("phases", json.dumps(meta["phases"]))
    print("jitter start", res["jitter0"])
    print("jitter end  ", res["jitter1"])
    for k, v in sens.items():
        print(f"  {k:16s} f {v['focal_px']:6.0f} dist {v['start_end_dist_h_cm']:5.1f} cm lift "
              f"{v['lift_max_cm']:5.2f} cm z_start {v['start_z_mm']:6.1f} z_end {v['end_z_mm']:6.1f} mm "
              f"tilt {v['start_tilt_deg']:.2f}/{v['end_tilt_deg']:.2f} yaw {v['yaw_change_deg']:.1f}")
    print(f"wrote {out / 'path.json'}, {out / 'path.png'}, {out / 'overlay.jpg'}")


if __name__ == "__main__":
    main()
