"""Scripted privileged-state controller for LIBERO-90 task 21
(KITCHEN_SCENE3: turn on the stove and put the frying pan on it).

Strategy "push-knob":
  * knob : gripper CLOSED, fingertips lowered next to the knob's grip ridge and
           swept tangentially around the knob axis (positive joint direction),
           closed-loop on the measured knob angle, until qpos >= KNOB_GOAL.
  * pan  : top-down grasp across the handle (fingers close perpendicular to the
           handle axis, default = handle middle). A clearance check against the
           other scene geoms (moka pot!) moves the grasp along the handle and/or
           deviates the yaw from perpendicular only when the default grasp would
           collide.
  * carry: pan poses are expressed in the STOVE frame (x, y, z, yaw of the pan
           origin). The gripper is servoed on the LIVE pan pose (object-centric
           P-control), so slip/sag inside the grasp is corrected and the same
           primitive can replay an arbitrary object path (e.g. extracted from a
           human phone video) given as a list of pan poses in the stove frame.

Primitives accept either the LIBERO env or a `Robot` wrapper (one wrapper is
cached per env so step budget / success bookkeeping are shared):
  turn_knob(env), grasp_pan_handle(env),
  carry_pan_to(env, target_pose_in_stove_frame, path=None),
  place_and_release(env, target_pose_in_stove_frame)

All geometry is expressed relative to object frames read from the simulator:
  knob axis        = body  flat_stove_1_button (hinge 'flat_stove_1_button', +z)
  knob ridge       = bar along the button +y/-y axis, near end at r~0.052 m,
                     half-width 0.012 m, top ~0.05 m above the stove origin
  cook region      = site  flat_stove_1_cook_region (= stove frame + [0.15,0,0])
  pan handle       = pan frame: axis +x, centre ~[0.19, 0, 0.044],
                     width (pan y) 0.025 m, thickness 0.012 m

Action convention: 7-D in [-1,1]; [0:3] eef translation (x0.05 m), [3:6]
axis-angle (x0.5 rad, world frame), [6] gripper (-1 open, +1 close).
"""
import numpy as np
import robosuite.utils.transform_utils as T

# ---------------------------------------------------------------- constants
# grip-site orientation of the "pointing straight down, yaw 0" pose (columns =
# site axes in world). Fingers close along site-x, i.e. world-y at yaw 0.
R_DOWN = np.array([[0.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0]])
OPEN, CLOSE = -1.0, 1.0

KNOB_BODY = "flat_stove_1_button"
KNOB_JOINT = "flat_stove_1_button"
STOVE = "flat_stove_1"
PAN = "chefmate_8_frypan_1"
COOK_SITE = "flat_stove_1_cook_region"

KNOB_GOAL = 0.85          # rad; success threshold is 0.5
KNOB_A = 0.030            # grip site distance along the ridge from the axis (m)
KNOB_DZ = 0.028           # grip-site height above the knob body origin
KNOB_B_START = -0.040     # lateral offset (ridge frame) before the push
KNOB_B_PUSH = 0.020       # lateral target while pushing (inside the ridge)

HANDLE_IN_PAN = np.array([0.185, 0.0, 0.044])   # grasp point, pan frame
HANDLE_AXIS_IN_PAN = np.array([1.0, 0.0, 0.0])
GRASP_DZ = 0.000          # grip site height offset from the handle centre

SAFE_Z_ABOVE_STOVE = 0.21  # pan-origin height above stove origin while carrying
HOVER_DZ = 0.04            # hover height above the place pose before lowering
# pan-origin place pose in the stove frame (x, y, z, yaw): cook region centre is
# (0.15, 0); y offset -0.035 keeps the arm out of its reach limit, yaw pi/2 =
# handle pointing along stove +y (away from the knob). Rest height 0.026.
PAN_TARGET_IN_STOVE = (0.15, -0.035, 0.026, np.pi / 2)


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of_quat_wxyz(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class EpisodeOver(Exception):
    pass


class Robot:
    """Thin wrapper: step budget, success bookkeeping, P-servo to eef targets."""

    def __init__(self, env, max_steps=400, on_step=None, stop_on_success=True,
                 verbose=False):
        self.env = env
        self.e = env.env
        self.sim = self.e.sim
        self.max_steps = max_steps
        self.on_step = on_step
        self.stop_on_success = stop_on_success
        self.verbose = verbose
        self.steps = 0
        self.success_step = None
        self.grip_sid = self.sim.model.site_name2id("gripper0_grip_site")
        self.log = []

    # ---- state accessors (privileged) ----
    def eef_pos(self):
        return self.sim.data.site_xpos[self.grip_sid].copy()

    def eef_mat(self):
        return self.sim.data.site_xmat[self.grip_sid].reshape(3, 3).copy()

    def eef_yaw(self):
        R = self.eef_mat() @ R_DOWN.T
        return np.arctan2(R[1, 0], R[0, 0])

    def body_pose(self, name):
        bid = self.sim.model.body_name2id(name) if name not in self.e.obj_body_id \
            else self.e.obj_body_id[name]
        return (self.sim.data.body_xpos[bid].copy(),
                self.sim.data.body_xquat[bid].copy())

    def body_mat(self, name):
        bid = self.sim.model.body_name2id(name) if name not in self.e.obj_body_id \
            else self.e.obj_body_id[name]
        return self.sim.data.body_xmat[bid].reshape(3, 3).copy()

    def knob_angle(self):
        adr = self.sim.model.get_joint_qpos_addr(KNOB_JOINT)
        return float(self.sim.data.qpos[adr])

    def gripper_width(self):
        adr0 = self.sim.model.get_joint_qpos_addr("gripper0_finger_joint1")
        adr1 = self.sim.model.get_joint_qpos_addr("gripper0_finger_joint2")
        return float(self.sim.data.qpos[adr0] - self.sim.data.qpos[adr1])

    def contact(self, a, b):
        return self.e.check_contact(self.e.get_object(a), self.e.get_object(b))

    # ---- stepping ----
    def step(self, action):
        if self.steps >= self.max_steps:
            raise EpisodeOver("step budget exhausted")
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        obs, _, _, _ = self.env.step(a)
        self.steps += 1
        if self.on_step is not None:
            self.on_step(obs)
        if self.success_step is None and self.env.check_success():
            self.success_step = self.steps
            if self.stop_on_success:
                raise EpisodeOver("success")
        return obs

    def action_to(self, pos, yaw, grip, gain=1.0, rot_gain=1.0, vmax=1.0):
        dp = (np.asarray(pos) - self.eef_pos()) / 0.05 * gain
        n = np.max(np.abs(dp))
        if n > vmax:
            dp = dp / n * vmax
        Rt = Rz(yaw) @ R_DOWN
        Rerr = Rt @ self.eef_mat().T
        aa = T.quat2axisangle(T.mat2quat(Rerr))
        dr = np.clip(aa / 0.5 * rot_gain, -1, 1)
        return np.concatenate([dp, dr, [grip]])

    def servo(self, pos, yaw, grip, tol=0.006, yaw_tol=0.06, max_steps=80,
              gain=1.0, vmax=1.0, min_steps=0):
        for k in range(max_steps):
            err = np.linalg.norm(np.asarray(pos) - self.eef_pos())
            yerr = abs(wrap(yaw - self.eef_yaw()))
            if k >= min_steps and err < tol and yerr < yaw_tol:
                return True
            self.step(self.action_to(pos, yaw, grip, gain=gain, vmax=vmax))
        return False

    def hold(self, grip, n, yaw=None, pos=None):
        pos = self.eef_pos() if pos is None else pos
        yaw = self.eef_yaw() if yaw is None else yaw
        for _ in range(n):
            self.step(self.action_to(pos, yaw, grip))


def as_robot(x, **kw):
    """Return the Robot wrapper for an env (cached on the env) or pass through."""
    if isinstance(x, Robot):
        return x
    rb = getattr(x, "_push_knob_robot", None)
    if rb is None:
        rb = Robot(x, **kw)
        x._push_knob_robot = rb
    return rb


# =================================================================== primitives
def turn_knob(rb, goal=KNOB_GOAL):
    """Closed gripper, finger cluster laid parallel to the knob ridge, pushes the
    ridge's near half sideways. The push target is expressed in the ROTATING
    ridge frame (re-evaluated from the measured hinge angle every step), so the
    fingertip sweeps tangentially around the knob axis until qpos >= goal."""
    rb = as_robot(rb)
    kc, _ = rb.body_pose(KNOB_BODY)
    _, sq = rb.body_pose(STOVE)
    stove_yaw = yaw_of_quat_wxyz(sq)
    z_push = kc[2] + KNOB_DZ

    def ridge_frame(theta):
        al = stove_yaw + np.pi + theta            # near end of the ridge
        u = np.array([np.cos(al), np.sin(al), 0.0])
        v = np.array([-np.sin(al), np.cos(al), 0.0])   # +joint motion direction
        # closing direction Rz(psi) e_y parallel to u  ->  psi = al - pi/2 (keeps joint 7 away from its limit)
        return u, v, wrap(al - np.pi / 2)

    def target(theta, b, dz=0.0):
        u, v, psi = ridge_frame(theta)
        return kc + KNOB_A * u + b * v + np.array([0, 0, z_push - kc[2] + dz]), psi

    th = rb.knob_angle()
    p, psi = target(th, KNOB_B_START, dz=0.07)
    rb.servo(p, psi, CLOSE, tol=0.01, yaw_tol=0.08, max_steps=70)
    p, psi = target(th, KNOB_B_START)
    rb.servo(p, psi, CLOSE, tol=0.005, yaw_tol=0.05, max_steps=40, vmax=0.6)
    for _ in range(100):
        th = rb.knob_angle()
        if th >= goal:
            break
        p, psi = target(th, KNOB_B_PUSH)
        rb.step(rb.action_to(p, psi, CLOSE, gain=1.0))
    if rb.verbose:
        print(f"  knob angle after push {rb.knob_angle():.3f} at step {rb.steps}")
    # retreat: step back off the ridge, then up
    th = rb.knob_angle()
    p, psi = target(th, KNOB_B_START * 0.5, dz=0.0)
    rb.servo(p, psi, CLOSE, tol=0.01, max_steps=8)
    rb.servo(p + np.array([0, 0, 0.08]), psi, CLOSE, tol=0.02, max_steps=25)
    return rb.knob_angle() >= 0.5


# open-gripper collision boxes in the grasp frame (c = closing dir, n = horizontal
# perpendicular, z = up from the grip site), measured from the Panda meshes.
GRIPPER_BOXES = [
    ((-0.100, 0.104), (-0.032, 0.032), (0.031, 0.30)),    # hand (+ swept descent)
    ((-0.066, -0.039), (-0.011, 0.011), (-0.009, 0.044)),  # finger 1 (open)
    ((0.039, 0.066), (-0.011, 0.011), (-0.009, 0.044)),    # finger 2 (open)
]
PREFERRED_YAW = 1.4       # grip-site yaw where Panda joint 7 is mid-range
GRASP_MARGIN = 0.010


def _obstacle_vertices(rb, exclude_root, exclude_geoms=()):
    """(world vertices, xy axes) of every collidable geom that is not part of the
    robot, the table or the object being grasped."""
    m, d = rb.sim.model, rb.sim.data
    M = m._model
    out = []
    for gid in range(M.ngeom):
        if M.geom_contype[gid] == 0 and M.geom_conaffinity[gid] == 0:
            continue
        b = M.geom_bodyid[gid]
        root = M.body_rootid[b]
        bname = m.body_id2name(root) or ""
        if b == 0 or bname.startswith(("robot", "mount", "gripper")) \
                or "table" in bname or "floor" in bname:
            continue
        if root == exclude_root and gid in exclude_geoms:
            continue
        t = M.geom_type[gid]
        R = d.geom_xmat[gid].reshape(3, 3)
        if t == 7:
            mid = M.geom_dataid[gid]
            a, n = M.mesh_vertadr[mid], M.mesh_vertnum[mid]
            v = M.mesh_vert[a:a + n]
            axes = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0])]
        else:
            s = M.geom_size[gid]
            if t != 6:          # sphere/capsule/cylinder: bounding box
                s = np.array([s[0], s[0], s[1] if t in (3, 5) else s[0]])
            v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                          for sz in (-1, 1)]) * s
            axes = []
            for k in range(3):
                ax = R[:, k].copy()
                ax[2] = 0.0
                if np.linalg.norm(ax) > 0.3:
                    axes.append(ax / np.linalg.norm(ax))
        out.append(((R @ v.T).T + d.geom_xpos[gid], axes))
    return out


def grasp_clearance(obstacles, g, yaw):
    """Separating-axis clearance (lower bound, m) between the open gripper at
    grip-site position g / yaw and the obstacle geoms."""
    c = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    n = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    boxes = []
    for (c0, c1), (n0, n1), (z0, z1) in GRIPPER_BOXES:
        corners = np.array([g + ci * c + ni * n + np.array([0, 0, zi])
                            for ci in (c0, c1) for ni in (n0, n1) for zi in (z0, z1)])
        boxes.append(corners)
    best = np.inf
    for V, oaxes in obstacles:
        for B in boxes:
            gap = max(V[:, 2].min() - B[:, 2].max(), B[:, 2].min() - V[:, 2].max())
            for ax in [c, n] + oaxes:
                pv, pb = V @ ax, B @ ax
                gap = max(gap, pv.min() - pb.max(), pb.min() - pv.max())
            best = min(best, gap)
    return best


def pan_handle_geoms(rb):
    """Collision geoms of the pan that form the handle bar (the grasp target):
    the long box whose centre is near HANDLE_IN_PAN."""
    M = rb.sim.model._model
    d = rb.sim.data
    pb = rb.e.obj_body_id[PAN]
    pp, Rp = d.body_xpos[pb], d.body_xmat[pb].reshape(3, 3)
    out = []
    for gid in range(M.ngeom):
        if M.geom_bodyid[gid] == pb and M.geom_contype[gid] and M.geom_type[gid] == 6:
            loc = Rp.T @ (d.geom_xpos[gid] - pp)
            if loc[0] > 0.15 and M.geom_size[gid].max() > 0.05:
                out.append(gid)
    return out


def handle_grasp_pose(rb, s_list=(0.185, 0.165, 0.205, 0.145, 0.225, 0.245),
                      beta_deg=tuple(range(-60, 61, 10))):
    """Top-down grasp across the pan handle. Default: middle of the handle, yaw
    perpendicular to the handle axis. If the hand would hit another object
    (e.g. the moka pot), search along the handle and over small yaw deviations
    for the closest collision-free alternative."""
    pp, pq = rb.body_pose(PAN)
    Rp = T.quat2mat(np.array([pq[1], pq[2], pq[3], pq[0]]))
    a = Rp @ HANDLE_AXIS_IN_PAN
    yaw_perp = np.arctan2(a[1], a[0])   # closing dir Rz(yaw) e_y is perp. to a
    obst = _obstacle_vertices(rb, rb.sim.model._model.body_rootid[rb.e.obj_body_id[PAN]],
                              exclude_geoms=pan_handle_geoms(rb))
    cands = []
    for s in s_list:
        h = pp + Rp @ np.array([s, HANDLE_IN_PAN[1], HANDLE_IN_PAN[2]])
        for bd in beta_deg:
            y = yaw_perp + np.deg2rad(bd)
            # equivalent yaw (mod pi) closest to the joint-7-friendly yaw
            y = y + np.pi * np.round((PREFERRED_YAW - y) / np.pi)
            clr = grasp_clearance(obst, h + np.array([0, 0, GRASP_DZ]), y)
            cost = abs(np.deg2rad(bd)) + 2.0 * abs(s - HANDLE_IN_PAN[0])
            cands.append((clr >= GRASP_MARGIN, -cost, clr, h, y, s, bd))
    ok = [c for c in cands if c[0]]
    best = max(ok, key=lambda c: c[1]) if ok else max(cands, key=lambda c: c[2])
    rb.grasp_choice = dict(s=best[5], beta_deg=best[6], clearance=best[2], feasible=bool(ok))
    return best[3], best[4]


def grasp_pan_handle(rb):
    """Approach above the chosen handle point, descend with the gripper open,
    close. Returns the grasp transform (pan pose in the gripper frame)."""
    rb = as_robot(rb)
    h, yaw = handle_grasp_pose(rb)
    if rb.verbose:
        print("  grasp choice", rb.grasp_choice)
    above = h + np.array([0, 0, 0.10])
    rb.servo(above, yaw, OPEN, tol=0.01, yaw_tol=0.08, max_steps=80)
    rb.servo(h + np.array([0, 0, 0.04]), yaw, OPEN, tol=0.006, max_steps=40, vmax=0.6)
    rb.servo(h + np.array([0, 0, GRASP_DZ]), yaw, OPEN, tol=0.004, max_steps=40,
             vmax=0.4)
    rb.hold(CLOSE, 12, yaw=yaw, pos=h + np.array([0, 0, GRASP_DZ]))
    # grasp transform: pan pose expressed in the gripper frame
    return measure_grasp(rb)


def measure_grasp(rb):
    rb = as_robot(rb)
    pp, pq = rb.body_pose(PAN)
    Rp = rb.body_mat(PAN)
    Rg, pg = rb.eef_mat(), rb.eef_pos()
    return dict(R=Rg.T @ Rp, p=Rg.T @ (pp - pg))


def stove_to_world(rb, pose_s):
    """pose_s = (x, y, z, yaw) of the pan origin in the stove frame."""
    sp, sq = rb.body_pose(STOVE)
    syaw = yaw_of_quat_wxyz(sq)
    p = sp + Rz(syaw) @ np.asarray(pose_s[:3])
    return p, wrap(syaw + pose_s[3])


def pan_yaw_world(rb):
    R = rb.body_mat(PAN)
    return np.arctan2(R[1, 0], R[0, 0])


def pan_servo_action(rb, pan_pos_w, pan_yaw_w, grip=CLOSE, vmax=1.0, dz=0.0):
    """One P-step that drives the HELD pan's measured pose towards the target.
    The gripper target is computed from the live pan pose (object-centric
    servoing), so slip/sag inside the grasp is corrected automatically."""
    pp = rb.body_pose(PAN)[0]
    g = rb.eef_pos()
    dyaw = wrap(pan_yaw_w - pan_yaw_world(rb))
    dyaw = float(np.clip(dyaw, -0.6, 0.6))
    rel = pp - g
    gp = np.asarray(pan_pos_w) - Rz(dyaw) @ rel + np.array([0, 0, dz])
    return rb.action_to(gp, rb.eef_yaw() + dyaw, grip, vmax=vmax)


def carry_pan_to(rb, target_pose_in_stove_frame, path=None,
                 via_tol=0.03, tol=0.012, max_steps_per_wp=60):
    """Move the grasped pan through `path` (list of (x,y,z,yaw) pan-origin poses
    in the STOVE frame) and end hovering above `target_pose_in_stove_frame`.
    When `path` is None a lift -> transfer -> hover path is generated.
    Intermediate waypoints are passed with a loose tolerance."""
    rb = as_robot(rb)
    sp, sq = rb.body_pose(STOVE)
    syaw = yaw_of_quat_wxyz(sq)
    tx, ty, tz, tyaw = target_pose_in_stove_frame
    if path is None:
        pp = rb.body_pose(PAN)[0]
        cur_s = Rz(-syaw) @ (pp - sp)
        cur_yaw = wrap(pan_yaw_world(rb) - syaw)
        zc = SAFE_Z_ABOVE_STOVE
        path = [
            (cur_s[0], cur_s[1], zc, cur_yaw),
            (0.5 * (cur_s[0] + tx), 0.5 * (cur_s[1] + ty), zc, tyaw),
            (tx, ty, zc, tyaw),
            (tx, ty, tz + HOVER_DZ, tyaw),
        ]
    for k, wp in enumerate(path):
        last = k == len(path) - 1
        pw, yw = stove_to_world(rb, wp)
        for _ in range(max_steps_per_wp):
            err = np.linalg.norm(rb.body_pose(PAN)[0] - pw)
            yerr = abs(wrap(yw - pan_yaw_world(rb)))
            if err < (tol if last else via_tol) and yerr < (0.08 if last else 0.3):
                break
            rb.step(pan_servo_action(rb, pw, yw, vmax=0.7 if last else 1.0))
    return stove_to_world(rb, target_pose_in_stove_frame)


def place_and_release(rb, target_pose_in_stove_frame=None, max_lower=50):
    """Lower the pan onto the target until it touches the stove, then open the
    gripper and retreat upwards."""
    rb = as_robot(rb)
    if target_pose_in_stove_frame is None:
        target_pose_in_stove_frame = PAN_TARGET_IN_STOVE
    pw, yw = stove_to_world(rb, target_pose_in_stove_frame)
    for _ in range(max_lower):
        rb.step(pan_servo_action(rb, pw, yw, vmax=0.4, dz=-0.01))
        if rb.contact(PAN, STOVE):
            break
    rb.hold(CLOSE, 2)
    p = rb.eef_pos()
    y = rb.eef_yaw()
    for _ in range(8):
        rb.step(rb.action_to(p, y, OPEN))
    rb.servo(p + np.array([0, 0, 0.08]), y, OPEN, tol=0.015, max_steps=30, vmax=0.6)
    rb.hold(OPEN, 10)


# ====================================================================== policy


def run_episode(env, max_steps=400, on_step=None, verbose=False, stop_on_success=True):
    rb = Robot(env, max_steps=max_steps, on_step=on_step, verbose=verbose,
               stop_on_success=stop_on_success)
    env._push_knob_robot = rb
    stage = "start"
    try:
        stage = "knob"
        turn_knob(rb)
        stage = "grasp"
        grasp = grasp_pan_handle(rb)
        if verbose:
            print(f"  grasp done step {rb.steps}, width {rb.gripper_width():.4f}, "
                  f"rel p {grasp['p'].round(3)}")
        stage = "lift"
        p = rb.eef_pos()
        rb.servo(p + np.array([0, 0, 0.05]), rb.eef_yaw(), CLOSE, tol=0.01,
                 max_steps=20)
        stage = "carry"
        carry_pan_to(rb, PAN_TARGET_IN_STOVE)
        stage = "place"
        place_and_release(rb, PAN_TARGET_IN_STOVE)
        stage = "idle"
        rb.release_step = rb.steps
        while True:
            rb.hold(OPEN, 1)
    except EpisodeOver as ex:
        if verbose:
            print(f"  episode over ({ex}) at stage {stage}, step {rb.steps}")
    return dict(success=rb.success_step is not None, success_step=rb.success_step,
                steps=rb.steps, stage=stage, knob=rb.knob_angle(),
                pan=rb.body_pose(PAN)[0], final_success=bool(env.check_success()),
                release_step=getattr(rb, "release_step", None),
                grasp=getattr(rb, "grasp_choice", None))
