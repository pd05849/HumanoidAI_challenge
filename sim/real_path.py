"""Real (phone-tracked) pan path -> scripted Franka replay in LIBERO-90 task 21
(KITCHEN_SCENE3, "turn on the stove and put the frying pan on it").

Used by sim/make_demos.py for every (take, layout) episode. The steps:

 1. Real path (scripts/track_pan.py path.json): board world frame in metres (x right
    along the counter, y toward the wall, z up; pan origin = centre of the marker on
    the pan bottom). Unmeasured frames are filled by time-based linear interpolation,
    then a zero-phase Savitzky-Golay filter (cubic, ~0.23 s window: 7 frames at 30 fps,
    15 at 60 fps) is applied to x, y, z and unwrapped yaw.
 2. Transport segment: lift-off - PRE ... touchdown + POST (the tracker's phases),
    resampled on the real clock at the 20 Hz control rate.
 3. Mapping, per layout, into the controller's STOVE frame (x away from the robot,
    y = robot's left, z up):
    xy : 2-D similarity (rotation + uniform scale + translation, never a reflection)
         real start rest -> sim pan pose right after the grasp, real end rest ->
         the place target PAN_TARGET_IN_STOVE (0.15, -0.035).
    z  : baseline blended from the sim start height to the place height with
         w = smoothstep(progress along the start->end chord), plus kz x the real lift
         above its own blended baseline (kz = 1: the real lift). The sim has a 15 cm
         moka pot beside the carry line that the real counter does not have: a
         kinematic pre-flight check (mj_geomDistance between the held pan and the
         moka pot's collision geoms at every waypoint) adds the smallest smooth local
         raise that gives >= `clearance` (lift="bump", default).
    yaw: sim start yaw + the real yaw change ("relative", default) or blended to end at
         the controller's place yaw ("blend").
    time: wherever the mapped pan would move faster than vmax (0.25 m/s) the clock is
         stretched locally; elsewhere real time is kept.
 4. Episode: turn_knob -> grasp_pan_handle -> carry_pan_to(place, path=..., via_tol=0,
    max_steps_per_wp=1) (one 20 Hz controller step per waypoint; the commanded
    waypoint is the desired one `lookahead` steps ahead, cancelling the ~0.2 s OSC
    servo lag) -> place_and_release(place) -> `idle` open-gripper steps.
"""
import json
from dataclasses import dataclass, asdict

import numpy as np

try:                                   # imported as sim.real_path or with sim/ on sys.path
    from . import kitchen_controller as C
except ImportError:                    # pragma: no cover
    import kitchen_controller as C

DT = 0.05            # LIBERO control period (20 Hz)
MOKA = "moka_pot_1"


# ============================================================================ helpers
def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def runs_of(mask):
    """[(start, end_inclusive)] of the True runs of a boolean array."""
    out, k, n = [], 0, len(mask)
    while k < n:
        if mask[k]:
            j = k
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((k, j))
            k = j + 1
        else:
            k += 1
    return out


# ============================================================================ 1. real path
REQUIRED_PHASES = ("rest_start_frames", "rest_end_frames", "lift_off_s", "touchdown_s")


def sg_frames(fps, window_s=0.23, order=3):
    """Odd Savitzky-Golay window (frames) covering ~window_s at this frame rate
    (7 at 29.97 fps, 15 at 59.94 fps)."""
    w = int(round(window_s * fps)) | 1
    return max(w, order + 2 + (order % 2 == 0))


def load_real(path_json, sg_window_s=0.23, sg_order=3):
    from scipy.signal import savgol_filter
    with open(path_json) as f:
        d = json.load(f)
    meta = d["metadata"]
    ph = meta.get("phases") or {}
    missing = [k for k in REQUIRED_PHASES if ph.get(k) is None]
    if missing:
        raise ValueError(f"{path_json}: metadata.phases lacks {missing}")
    rows = d["path"]
    n = len(rows)
    t = np.array([r["t"] for r in rows], float)
    fps = float(meta.get("fps") or 1.0 / np.median(np.diff(t)))
    raw = np.full((n, 4), np.nan)
    meas = np.zeros(n, bool)
    for k, r in enumerate(rows):
        ok = bool(r.get("measured", r.get("detected", False))) and \
            all(r.get(c) is not None for c in ("x", "y", "z", "yaw"))
        if ok:
            raw[k] = [r["x"], r["y"], r["z"], r["yaw"]]
            meas[k] = True
    if meas.sum() < 10:
        raise ValueError(f"{path_json}: fewer than 10 measured frames")
    raw[meas, 3] = np.unwrap(raw[meas, 3])
    filled = raw.copy()
    for c in range(4):
        filled[~meas, c] = np.interp(t[~meas], t[meas], raw[meas, c])
    gaps = [(int(a), int(b)) for a, b in runs_of(~meas)]
    sg_win = sg_frames(fps, sg_window_s, sg_order)
    smooth = savgol_filter(filled, sg_win, sg_order, axis=0, mode="interp")
    cmp = None
    if "path_smoothed" in d:
        ts = np.array([[p["x"], p["y"], p["z"]] for p in d["path_smoothed"]], float)
        if len(ts) == n and np.all(np.isfinite(ts)):
            diff = np.linalg.norm(smooth[:, :3] - ts, axis=1)
            cmp = dict(rms_mm=float(np.sqrt(np.mean(diff ** 2)) * 1e3), max_mm=float(diff.max() * 1e3))
    resid = np.linalg.norm(smooth[meas, :3] - raw[meas, :3], axis=1)
    rs = ph["rest_start_frames"]
    re_ = ph["rest_end_frames"]
    start = filled[rs[0]:rs[1] + 1].mean(0)
    end = filled[re_[0]:re_[1] + 1].mean(0)
    if np.hypot(*(end[:2] - start[:2])) < 0.05:
        raise ValueError(f"{path_json}: start and end rest poses are < 5 cm apart")
    if not ph["lift_off_s"] < ph["touchdown_s"]:
        raise ValueError(f"{path_json}: lift_off_s >= touchdown_s")
    return dict(t=t, raw=raw, filled=filled, smooth=smooth, measured=meas, gaps=gaps,
                n_frames=n, n_measured=int(meas.sum()), phases=ph, start=start, end=end, fps=fps,
                smooth_residual_rms_mm=float(np.sqrt(np.mean(resid ** 2)) * 1e3),
                smooth_vs_tracker=cmp, sg=(sg_win, sg_order), video=meta.get("video"),
                low_confidence_frames=meta.get("detection", {}).get("ecc_correlation_below_0p92", []))


def real_summary(R):
    ph = R["phases"]
    return dict(n_frames=R["n_frames"], n_measured=R["n_measured"], fps=R["fps"], sg=list(R["sg"]),
                gaps=R["gaps"], start_cm=(R["start"][:3] * 100).round(2).tolist(),
                end_cm=(R["end"][:3] * 100).round(2).tolist(),
                travel_cm=float(np.hypot(*(R["end"][:2] - R["start"][:2])) * 100),
                yaw_change_deg=float(np.degrees(R["end"][3] - R["start"][3])),
                lift_above_baseline_cm=kz_peak_real(R) * 100,
                lift_off_s=ph["lift_off_s"], touchdown_s=ph["touchdown_s"],
                carry_s=float(ph["touchdown_s"] - ph["lift_off_s"]),
                smooth_residual_rms_mm=R["smooth_residual_rms_mm"], smooth_vs_tracker=R["smooth_vs_tracker"])


# ============================================================================ 2. segment
def real_segment(R, pre=0.3, post=0.3, speedup=1.0):
    """The transport window resampled on the real clock, one sample per 20 Hz control step
    (speedup > 1 plays the real motion faster: one step covers speedup x DT of real time)."""
    ph, t = R["phases"], R["t"]
    t0 = max(t[0], ph["lift_off_s"] - pre)
    t1 = min(t[-1], ph["touchdown_s"] + post)
    tk = np.arange(t0, t1 + 1e-9, DT * float(speedup))
    P = np.column_stack([np.interp(tk, t, R["smooth"][:, c]) for c in range(4)])
    return tk, P


def kz_peak_real(R):
    """Peak of the real lift above its own blended baseline (the quantity kz scales)."""
    tk, P = real_segment(R, 0.0, 0.0)
    s_r, e_r = R["start"], R["end"]
    zr = P[:, 0] + 1j * P[:, 1]
    u = np.real((zr - complex(*s_r[:2])) / (complex(*e_r[:2]) - complex(*s_r[:2])))
    w = smoothstep(u)
    return float(np.max(P[:, 2] - (s_r[2] + w * (e_r[2] - s_r[2]))))


# ============================================================================ 3. mapping
def similarity(R, start_s, target_s):
    zs_r, ze_r = complex(*R["start"][:2]), complex(*R["end"][:2])
    zs_s, ze_s = complex(*start_s[:2]), complex(*target_s[:2])
    return (ze_s - zs_s) / (ze_r - zs_r), zs_r, ze_r, zs_s, ze_s


def map_segment(tk, P, R, start_s, target_s, kz=1.0, yaw_mode="relative"):
    """Real segment P (N x [x, y, z, yaw], board frame) -> waypoints (N x [x, y, z, yaw], stove frame)."""
    s_r, e_r = R["start"], R["end"]
    a, zs_r, ze_r, zs_s, ze_s = similarity(R, start_s, target_s)
    zr = P[:, 0] + 1j * P[:, 1]
    zm = zs_s + a * (zr - zs_r)
    u = np.real((zr - zs_r) / (ze_r - zs_r))              # progress along the real chord
    w = smoothstep(u)
    lift_real = P[:, 2] - (s_r[2] + w * (e_r[2] - s_r[2]))  # real lift above its own baseline
    base_sim = start_s[2] + w * (target_s[2] - start_s[2])
    z = base_sim + kz * lift_real
    dyaw = P[:, 3] - s_r[3]
    dyaw_total = e_r[3] - s_r[3]
    if yaw_mode == "blend":
        place_yaw = target_s[3]
        yaw = start_s[3] + dyaw - w * (dyaw_total - wrap(place_yaw - start_s[3]))
    elif yaw_mode == "relative":
        place_yaw = start_s[3] + dyaw_total
        yaw = start_s[3] + dyaw
    else:
        raise ValueError(yaw_mode)
    W = np.column_stack([zm.real, zm.imag, z, yaw])
    place = (float(target_s[0]), float(target_s[1]), float(target_s[2]), float(place_yaw))
    return W, dict(a=a, w=w, u=u, lift_real=lift_real, base_sim=base_sim, place=place)


def speed_cap(W, tk, vmax):
    """Re-time the waypoints (one per 20 Hz step) so that no step moves the pan origin faster
    than vmax. Returns the re-timed waypoints and, for every one, the real-clock time it came from."""
    if vmax is None or vmax <= 0:
        return W.copy(), tk.copy()
    ds = np.linalg.norm(np.diff(W[:, :3], axis=0), axis=1)
    dt = np.maximum(DT, ds / vmax)
    tn = np.r_[0.0, np.cumsum(dt)]
    tq = np.arange(0.0, tn[-1] + 1e-9, DT)
    if tn[-1] - tq[-1] > 1e-6:
        tq = np.r_[tq, tq[-1] + DT]
    Wq = np.column_stack([np.interp(tq, tn, W[:, c]) for c in range(4)])
    return Wq, np.interp(tq, tn, tk)


# ============================================================================ sim helpers
def pan_in_stove(rb):
    sp, sq = rb.body_pose(C.STOVE)
    syaw = C.yaw_of_quat_wxyz(sq)
    p = C.Rz(-syaw) @ (rb.body_pose(C.PAN)[0] - sp)
    return np.array([p[0], p[1], p[2], C.wrap(C.pan_yaw_world(rb) - syaw)])


def to_stove(rb, p_world):
    sp, sq = rb.body_pose(C.STOVE)
    return C.Rz(-C.yaw_of_quat_wxyz(sq)) @ (np.asarray(p_world) - sp)


def moka_state(rb):
    p, _ = rb.body_pose(MOKA)
    R = rb.body_mat(MOKA)
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    return p, float(tilt)


class Geo:
    """Geom bookkeeping: which geoms belong to the pan, the moka pot, the stove, the robot."""

    def __init__(self, rb):
        import mujoco
        self.mj = mujoco
        M = rb.sim.model._model
        self.M = M
        e = rb.e
        root = M.body_rootid
        self.pan_root = root[e.obj_body_id[C.PAN]]
        self.moka_root = root[e.obj_body_id[MOKA]]
        self.stove_root = root[e.obj_body_id[C.STOVE]]
        collid = (M.geom_contype != 0) | (M.geom_conaffinity != 0)
        groot = root[M.geom_bodyid]
        self.pan_geoms = np.where(collid & (groot == self.pan_root))[0]
        self.moka_geoms = np.where(collid & (groot == self.moka_root))[0]
        names = [mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, int(b)) or "" for b in M.geom_bodyid]
        self.robot_geom = np.array([nm.startswith(("robot0", "gripper0")) for nm in names])
        self.groot = groot
        j = [j for j in range(M.njnt) if M.jnt_bodyid[j] == self.pan_root and M.jnt_type[j] == 0]
        self.pan_qadr = int(M.jnt_qposadr[j[0]])

    def contacts(self, sim):
        d = sim.data._data
        out = dict(pan_moka=False, robot_moka=False, pan_stove=False)
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            r1, r2 = self.groot[g1], self.groot[g2]
            pair = {r1, r2}
            if pair == {self.pan_root, self.moka_root}:
                out["pan_moka"] = True
            elif self.moka_root in pair and (self.robot_geom[g1] or self.robot_geom[g2]):
                out["robot_moka"] = True
            elif pair == {self.pan_root, self.stove_root}:
                out["pan_stove"] = True
        return out

    @staticmethod
    def _witness_ok(M, d, g1, g2, dd, ft):
        """MuJoCo's box-box distance occasionally returns a spurious 0; accept a distance only
        if both witness points lie within their geom's bounding sphere and the segment
        matches |dd|."""
        if not np.all(np.isfinite(ft)):
            return False
        a, b = ft[:3], ft[3:]
        if np.linalg.norm(a - d.geom_xpos[g1]) > M.geom_rbound[g1] + 1e-3:
            return False
        if np.linalg.norm(b - d.geom_xpos[g2]) > M.geom_rbound[g2] + 1e-3:
            return False
        return abs(np.linalg.norm(b - a) - abs(dd)) < 2e-3

    def preflight(self, rb, W, margin_dist=0.25):
        """Minimum distance (m) between the pan (rigid, at its current tilt, yaw following the
        waypoints) and the moka pot, for every waypoint (stove frame). Kinematic only, on a
        scratch MjData: the live simulation is not touched."""
        mujoco = self.mj
        M = self.M
        D = rb.sim.data._data
        d2 = mujoco.MjData(M)
        d2.qpos[:] = D.qpos
        mujoco.mj_kinematics(M, d2)
        q0 = D.qpos[self.pan_qadr + 3:self.pan_qadr + 7].copy()     # wxyz
        yaw0 = C.pan_yaw_world(rb)
        sp, sq = rb.body_pose(C.STOVE)
        syaw = C.yaw_of_quat_wxyz(sq)
        out = np.empty(len(W))
        ft = np.zeros(6)
        for k, wp in enumerate(W):
            pw = sp + C.Rz(syaw) @ np.asarray(wp[:3])
            dpsi = (syaw + wp[3]) - yaw0
            qz = np.array([np.cos(dpsi / 2), 0.0, 0.0, np.sin(dpsi / 2)])
            q = np.empty(4)
            mujoco.mju_mulQuat(q, qz, q0)
            d2.qpos[self.pan_qadr:self.pan_qadr + 3] = pw
            d2.qpos[self.pan_qadr + 3:self.pan_qadr + 7] = q
            mujoco.mj_kinematics(M, d2)
            best = margin_dist
            for g1 in self.pan_geoms:
                for g2 in self.moka_geoms:
                    ft[:] = np.nan
                    dd = mujoco.mj_geomDistance(M, d2, int(g1), int(g2), margin_dist, ft)
                    if dd < best and self._witness_ok(M, d2, int(g1), int(g2), dd, ft):
                        best = dd
            out[k] = best
        return out

    def moka_footprint(self, rb):
        """Axis-aligned footprint [[x0, x1], [y0, y1]] and top height of the moka pot's
        collision geoms, in the stove frame."""
        M, D = rb.sim.model._model, rb.sim.data._data
        verts = []
        for g in self.moka_geoms:
            if M.geom_type[g] == 7:
                mid = M.geom_dataid[g]
                a0, nv = M.mesh_vertadr[mid], M.mesh_vertnum[mid]
                v = M.mesh_vert[a0:a0 + nv]
            else:
                s = M.geom_size[g]
                v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * s
            vw = (D.geom_xmat[g].reshape(3, 3) @ v.T).T + D.geom_xpos[g]
            verts.append(np.array([to_stove(rb, p) for p in vw]))
        V = np.vstack(verts)
        return ([[float(V[:, 0].min()), float(V[:, 0].max())], [float(V[:, 1].min()), float(V[:, 1].max())]],
                float(V[:, 2].max()))


def required_raise(geo, rb, W, margin, dmax=0.40, iters=12):
    """Per waypoint, the smallest vertical raise (m) of the pan that gives >= margin clearance to
    the moka pot (0 where it already has it). Each waypoint is independent (kinematic check)."""
    c0 = geo.preflight(rb, W)
    need = np.zeros(len(W))
    for k in np.where(c0 < margin)[0]:
        lo, hi = 0.0, dmax
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            w2 = W[k:k + 1].copy()
            w2[0, 2] += mid
            if geo.preflight(rb, w2)[0] >= margin:
                hi = mid
            else:
                lo = mid
        need[k] = hi
    return need, c0


def raise_envelope(need, half_window=5, sigma=2.0):
    """Smooth upper envelope of the per-waypoint raise: widened by +-half_window samples, Gaussian
    smoothed, never below the requirement."""
    from scipy.ndimage import maximum_filter1d, gaussian_filter1d
    if need.max() <= 0:
        return need.copy()
    b = maximum_filter1d(need, size=2 * half_window + 1, mode="nearest")
    b = gaussian_filter1d(b, sigma, mode="nearest")
    return np.maximum(b, need)


# ============================================================================ 4. episode
@dataclass
class ReplayConfig:
    lift: str = "bump"          # bump | real | scale
    kz: float = 1.0             # base lift scale (1 = the real lift)
    kz_max: float = 3.0
    clearance: float = 0.02     # pre-flight pan-moka clearance (m) for lift bump/scale
    vmax: float = 0.25          # speed cap for the pan origin (m/s); 0 = none
    lookahead: int = 4          # command the desired pose this many 20 Hz steps ahead
    yaw: str = "relative"       # relative | blend
    pre: float = 0.3            # rest before lift-off to replay (s)
    post: float = 0.3           # rest after touchdown to replay (s)
    idle: int = 15              # open-gripper steps after the release
    speedup: float = 1.0        # real-time playback factor of the carry (1 = real time)

    def as_dict(self):
        return asdict(self)


def plan_carry(rb, geo, R, start_s, cfg):
    """Map the real transport segment onto this layout (stove frame), add the moka-pot
    clearance raise, apply the speed cap. Returns the waypoint plan."""
    target = C.PAN_TARGET_IN_STOVE
    tk, P = real_segment(R, cfg.pre, cfg.post, cfg.speedup)
    bump = np.zeros(len(tk))
    info = {}
    kz_used = float(cfg.kz)
    Wm0, _ = map_segment(tk, P, R, start_s, target, kz=kz_used, yaw_mode=cfg.yaw)
    c_real = geo.preflight(rb, Wm0)
    info["preflight_min_clearance_real_lift_m"] = float(c_real.min())
    if cfg.lift == "scale" and c_real.min() < cfg.clearance:
        kz_lo, kz_hi = kz_used, cfg.kz_max
        for _ in range(12):
            mid = 0.5 * (kz_lo + kz_hi)
            Wt, _ = map_segment(tk, P, R, start_s, target, kz=mid, yaw_mode=cfg.yaw)
            if geo.preflight(rb, Wt).min() >= cfg.clearance:
                kz_hi = mid
            else:
                kz_lo = mid
        kz_used = kz_hi
    elif cfg.lift == "bump" and c_real.min() < cfg.clearance:
        need, _ = required_raise(geo, rb, Wm0, cfg.clearance)
        bump = raise_envelope(need)
        for _ in range(3):        # the envelope also moves neighbours: re-check, top up
            Wb = Wm0.copy()
            Wb[:, 2] += bump
            extra, _ = required_raise(geo, rb, Wb, cfg.clearance - 1e-3)
            if extra.max() <= 0:
                break
            bump = raise_envelope(bump + extra)
        info.update(raise_max_cm=float(bump.max() * 100),
                    raise_real_time_s=[float(tk[np.argmax(bump > 1e-4)]),
                                       float(tk[len(bump) - 1 - np.argmax(bump[::-1] > 1e-4)])],
                    raise_fraction=float(np.mean(bump > 1e-4)))
    Wm, mi = map_segment(tk, P, R, start_s, target, kz=kz_used, yaw_mode=cfg.yaw)
    Wm[:, 2] += bump
    W, treal = speed_cap(Wm, tk, cfg.vmax)
    bump_W = np.interp(treal, tk, bump)
    clearance = geo.preflight(rb, W)
    a = mi["a"]
    place = mi["place"]
    # sanity: real 'toward the wall' (+y_real) -> stove frame direction (+x = away from the robot)
    back = a * 1j / abs(a)
    base_s = to_stove(rb, rb.sim.data.body_xpos[rb.sim.model.body_name2id("robot0_base")])
    mid_xy = 0.5 * (np.asarray(start_s[:2]) + np.asarray(place[:2]))
    away = (mid_xy - base_s[:2]) / np.linalg.norm(mid_xy - base_s[:2])
    info.update(scale=float(abs(a)), rotation_deg=float(np.degrees(np.angle(a))),
                back_of_counter_dir_deg=float(np.degrees(np.angle(back))),
                back_of_counter_dot_away=float(back.real * away[0] + back.imag * away[1]),
                kz=float(kz_used), lift_mode=cfg.lift, n_waypoints=int(len(W)),
                duration_s=float(len(W) * DT), real_window_s=[float(tk[0]), float(tk[-1])],
                real_duration_s=float(tk[-1] - tk[0]),
                peak_speed_cmd=float(np.max(np.linalg.norm(np.diff(W[:, :3], axis=0), axis=1)) / DT),
                peak_speed_uncapped=float(np.max(np.linalg.norm(np.diff(Wm[:, :3], axis=0), axis=1)) / DT),
                start_s=np.asarray(start_s).tolist(), place=list(place),
                preflight_min_clearance_m=float(clearance.min()))
    return dict(W=W, treal=treal, bump_W=bump_W, place=place, clearance=clearance, info=info)


class StepLogger:
    """rb.on_step hook: per control step, pan pose (stove frame), eef, knob, moka pose,
    contacts and the current phase. Increments the carry waypoint counter."""

    def __init__(self, rb, geo, phase):
        self.rb, self.geo, self.phase = rb, geo, phase
        self.log = []

    def __call__(self, _obs):
        rb = self.rb
        mp, mt = moka_state(rb)
        rec = dict(step=rb.steps, phase=self.phase["name"], k=self.phase["k"],
                   pan=pan_in_stove(rb).tolist(), eef=to_stove(rb, rb.eef_pos()).tolist(),
                   knob=rb.knob_angle(), moka=mp.tolist(), moka_tilt=mt,
                   success=bool(rb.success_step is not None), **self.geo.contacts(rb.sim))
        self.log.append(rec)
        if self.phase["name"] == "carry" and self.phase["k"] is not None:
            self.phase["k"] += 1


def replay(rb, R, cfg, phase=None, geo=None):
    """One full episode on the current layout: turn knob, grasp the handle, carry the pan
    along the mapped real path, place, release, idle. Uses rb.on_step if set.
    `phase` (dict with 'name', 'k') is updated as the episode advances."""
    phase = phase if phase is not None else {}
    phase.update(name="knob", k=None)
    geo = geo or Geo(rb)
    init_pan = pan_in_stove(rb)
    marks, info = {}, {}
    plan = None
    status = "ok"
    try:
        C.turn_knob(rb)
        marks["knob_done"] = rb.steps
        phase["name"] = "grasp"
        C.grasp_pan_handle(rb)
        marks["grasp_done"] = rb.steps
        info["grasp_choice"] = getattr(rb, "grasp_choice", None)
        g0 = C.measure_grasp(rb)
        start_s = pan_in_stove(rb)
        plan = plan_carry(rb, geo, R, start_s, cfg)
        info.update(plan["info"])
        info.update(init_pan_s=init_pan.tolist(),
                    grasp_moved_pan_mm=float(np.linalg.norm(start_s[:3] - init_pan[:3]) * 1e3))
        W = plan["W"]
        L = int(cfg.lookahead)
        cmd = [tuple(W[min(k + L, len(W) - 1)]) for k in range(len(W))]
        phase["name"], phase["k"] = "carry", 0
        marks["carry_start"] = rb.steps
        C.carry_pan_to(rb, plan["place"], path=cmd, via_tol=0.0, max_steps_per_wp=1)
        marks["carry_done"] = rb.steps
        phase["k"] = None
        g1 = C.measure_grasp(rb)
        info["grasp_slip_mm"] = float(np.linalg.norm(g1["p"] - g0["p"]) * 1e3)
        info["grasp_slip_deg"] = float(np.degrees(np.arccos(np.clip(
            (np.trace(g1["R"].T @ g0["R"]) - 1) / 2, -1, 1))))
        phase["name"] = "place"
        C.place_and_release(rb, plan["place"])
        marks["released"] = rb.steps
        phase["name"] = "idle"
        rb.hold(C.OPEN, cfg.idle)
    except C.EpisodeOver as ex:
        status = f"episode over: {ex}"
    return dict(status=status, marks=marks, info=info, plan=plan, geo=geo)


def carry_metrics(log, plan):
    """Tracking of the commanded path during the carry. The pan pose logged after the k-th carry
    step is at time (k+1) DT on the waypoint clock, so it is compared with W[k+1]."""
    if plan is None:
        return None
    W = plan["W"]
    car = [r for r in log if r["phase"] == "carry"]
    if len(car) < 3:
        return None
    A = np.array([r["pan"] for r in car])
    n = len(A)
    idx = np.minimum(np.arange(1, n + 1), len(W) - 1)
    Dsync = W[idx]
    e3 = np.linalg.norm(A[:, :3] - Dsync[:, :3], axis=1)
    eyaw = np.degrees(wrap(A[:, 3] - Dsync[:, 3]))
    seg_a, seg_b = W[:-1, :3], W[1:, :3]
    ab = seg_b - seg_a
    ab2 = np.maximum(np.sum(ab ** 2, axis=1), 1e-12)
    dpath = []
    for p in A[:, :3]:
        tt = np.clip(np.sum((p - seg_a) * ab, axis=1) / ab2, 0, 1)
        q = seg_a + tt[:, None] * ab
        dpath.append(np.min(np.linalg.norm(q - p, axis=1)))
    dpath = np.array(dpath)
    return dict(n_steps=int(n), rms_sync_3d_cm=float(np.sqrt(np.mean(e3 ** 2)) * 100),
                max_sync_3d_cm=float(e3.max() * 100), rms_yaw_deg=float(np.sqrt(np.mean(eyaw ** 2))),
                rms_path_shape_cm=float(np.sqrt(np.mean(dpath ** 2)) * 100),
                max_path_shape_cm=float(dpath.max() * 100),
                pan_peak_z_cm=float(A[:, 2].max() * 100), cmd_peak_z_cm=float(W[:, 2].max() * 100))
