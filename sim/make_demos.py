#!/usr/bin/env python
"""Convert tracked real pan paths (scripts/track_pan.py -> path.json) into LIBERO robot demos
for LIBERO-90 task 21 "turn on the stove and put the frying pan on it" (KITCHEN_SCENE3).

usage (LIBERO venv, Python 3.11):
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python sim/make_demos.py \
      TRACKS_DIR_OR_PATH_JSON [...] --layouts 3 --seed 0 --out OUT [--workers 2] \
      [--states-from sampled|eval] [--skip-existing]

Per (take, layout) episode:
  * Layout. --states-from sampled (default): a FRESH env is seeded (env.seed(s), i.e. numpy's
    global RNG, which LIBERO's BDDL region samplers use) and hard-reset, so LIBERO samples a
    new placement of the pan, moka pot and stove from the task's BDDL regions; the robot's
    start joints get N(0, 0.02 rad) noise like the 50 eval init states (their empirical
    spread is 0.021 rad per joint). The layout is the flattened MuJoCo state right after
    that reset. Its distance to the 50 fixed eval init states (pan and moka xy) is reported
    (and can be enforced with --min-eval-dist). --states-from eval uses eval init state
    (seed mod 50) instead (for comparison runs only: those are the test layouts).
    NB the stove is a LIBERO fixture: its pose is set in the model at reset (seeded), it is
    not part of any init-state vector, at eval time either.
  * Exactly the LeRobot eval reset: env.set_init_state(state), then 10 dummy steps
    [0,0,0,0,0,0,-1] (LiberoEnv.num_steps_wait); the first recorded frame is the obs after them.
  * Scripted replay (sim/real_path.py): turn knob, grasp the handle, carry the pan along the
    real path mapped onto this layout, place, release, idle. Success = env.check_success()
    at the end. Only successful episodes are written.
  * Prescreen (default on): the episode is first simulated without cameras (fast, ~10 s); only
    if it succeeds is it re-simulated with the 256x256 cameras on and recorded (same seed, same
    closed-loop controller; the recorded run's own success is what decides; the metadata says
    whether both runs sent identical actions). Rendering dominates: ~0.2-0.4 s per step.
  * --layouts N = successful episodes wanted per take: a failed layout (typically the moka pot
    so close to the pan handle that no collision-free grasp exists, ~20 % of sampled layouts)
    is replaced by the next seed, up to --max-attempts per take. Each worker needs ~2.3 GB RAM.

Recorded at every control step, in LeRobot LIBERO conventions (what the policy sees and does):
  image  (T,256,256,3) uint8  obs["agentview_image"][::-1, ::-1]          -> observation.images.image
  image2 (T,256,256,3) uint8  obs["robot0_eye_in_hand_image"][::-1, ::-1] -> observation.images.image2
         (rendered by the env at camera_heights/widths 256, rotated 180 deg as LiberoProcessorStep does)
  state  (T,8) float32  [eef_pos(3), axis-angle(eef_quat xyzw)(3), gripper_qpos(2)]  -> observation.state
         (axis-angle computed exactly like LiberoProcessorStep._quat2axisangle, in float32)
  action (T,7) float32  the 7-D action sent to env.step (clipped to [-1,1]; gripper -1 open, +1 close)
  step   (T,)  int32    control step index (0 = first step after the 10 settle steps)
  phase  (T,)  int8     0 knob, 1 grasp, 2 carry, 3 place, 4 idle
  + init_state (flattened MuJoCo state), stove_pos, stove_quat (the fixture pose of this layout)
  task string = "turn on the stove and put the frying pan on it", fps = 20.
  Frame t pairs the observation BEFORE action t with action t. No-op frames are dropped
  as in the OpenVLA/LeRobot LIBERO data: |action[:6]| < 1e-4 and gripper equal to the
  previous kept action's. Metadata: first_success_step = control step of the action after
  which check_success() first held, first_success_frame = its index among the kept frames
  (the recording continues through the release and idle steps after it).

Output (OUT/):
  episodes/<take>_seed<S>.npz + .json   successful episodes (arrays above; metadata + trace)
  failed/<take>_seed<S>.json            failed episodes (metadata only)
  run.json                              arguments, per-take real-path summary, per-episode results
  first_episode.mp4                     agentview of the first successful episode (20 fps, as stored)
  paths.png                             commanded (mapped real) vs achieved sim pan path, top view
                                        + height, for up to 6 episodes
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import zlib

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import real_path as RP                    # noqa: E402  (imports kitchen_controller)
import kitchen_controller as C            # noqa: E402

SUITE, TASK_ID = "libero_90", 21
TASK = "turn on the stove and put the frying pan on it"
CAM = 256
FPS = 20
DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1.0])     # lerobot get_libero_dummy_action()
INIT_NOISE = {"magnitude": 0.02, "type": "gaussian"}  # robot start-joint noise of the eval states
PHASES = ("knob", "grasp", "carry", "place", "idle")
PAN_JOINT, MOKA_JOINT = "chefmate_8_frypan_1_joint0", "moka_pot_1_joint0"
ROBOT_Q0 = np.array([0.0, -0.161, 0.0, -2.4446, 0.0, 2.2268, 0.7854])   # LIBERO Panda init_qpos


# ============================================================================ conventions
def quat2axisangle_lerobot(q_xyzw):
    """LiberoProcessorStep._quat2axisangle (float32): xyzw quaternion -> axis * angle."""
    q = np.asarray(q_xyzw, dtype=np.float32)
    w = np.clip(q[3], np.float32(-1.0), np.float32(1.0))
    den = np.sqrt(np.maximum(np.float32(1.0) - w * w, np.float32(0.0)))
    if den > 1e-10:
        return (q[:3] / den) * (np.float32(2.0) * np.arccos(w))
    return np.zeros(3, np.float32)


def lerobot_state(obs):
    """observation.state = [eef_pos(3), axis-angle(eef_quat)(3), gripper_qpos(2)], float32."""
    return np.concatenate([np.asarray(obs["robot0_eef_pos"], np.float64),
                           quat2axisangle_lerobot(obs["robot0_eef_quat"]).astype(np.float64),
                           np.asarray(obs["robot0_gripper_qpos"], np.float64)]).astype(np.float32)


def is_noop(action, prev_action, threshold=1e-4):
    """OpenVLA regenerate_libero_dataset.py no-op rule."""
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1]


class RecordingRobot(C.Robot):
    """kitchen_controller.Robot that records, for every control step, the observation the
    action was computed from and the action actually sent."""

    def __init__(self, env, obs0, images, phase, **kw):
        super().__init__(env, **kw)
        self.obs = obs0
        self.images = images
        self.phase = phase
        self.rec = dict(state=[], action=[], phase=[], image=[], image2=[])

    def step(self, action):
        if self.steps >= self.max_steps:
            raise C.EpisodeOver("step budget exhausted")
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        r = self.rec
        r["state"].append(lerobot_state(self.obs))
        r["action"].append(a.astype(np.float32))
        r["phase"].append(PHASES.index(self.phase["name"]))
        if self.images:
            r["image"].append(np.ascontiguousarray(self.obs["agentview_image"][::-1, ::-1]))
            r["image2"].append(np.ascontiguousarray(self.obs["robot0_eye_in_hand_image"][::-1, ::-1]))
        obs = super().step(a)
        self.obs = obs
        return obs


# ============================================================================ env / layouts
_CACHE = {}


def libero_task():
    if "task" not in _CACHE:
        from libero.libero import benchmark, get_libero_path
        suite = benchmark.get_benchmark_dict()[SUITE]()
        task = suite.get_task(TASK_ID)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        _CACHE["task"] = (bddl, task.language, np.asarray(suite.get_task_init_states(TASK_ID)))
    return _CACHE["task"]


def make_env(images):
    from libero.libero.envs import OffScreenRenderEnv
    bddl, _, _ = libero_task()
    return OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=CAM, camera_widths=CAM,
                              use_camera_obs=images, initialization_noise=INIT_NOISE)


def layout_of(env, init):
    """Pan / moka xy of the init-state vector (pre-settle, as sampled), the stove fixture pose,
    the robot's start-joint offset."""
    sim = env.env.sim
    pa = 1 + sim.model.get_joint_qpos_addr(PAN_JOINT)[0]
    ma = 1 + sim.model.get_joint_qpos_addr(MOKA_JOINT)[0]
    bid = env.env.obj_body_id[C.STOVE]
    M = sim.model._model
    return dict(pan_xy=init[pa:pa + 2].tolist(), moka_xy=init[ma:ma + 2].tolist(),
                stove_pos=M.body_pos[bid].tolist(), stove_quat=M.body_quat[bid].tolist(),
                robot_dq=(init[1:8] - ROBOT_Q0).tolist(), _adr=(pa, ma))


def eval_distance(layout, evals):
    pa, ma = layout["_adr"]
    dp = np.linalg.norm(evals[:, pa:pa + 2] - np.asarray(layout["pan_xy"]), axis=1)
    dm = np.linalg.norm(evals[:, ma:ma + 2] - np.asarray(layout["moka_xy"]), axis=1)
    dj = np.hypot(dp, dm)
    i = int(np.argmin(dj))
    return dict(min_joint_cm=float(dj[i] * 100), nearest_eval_state=i,
                pan_cm_at_nearest=float(dp[i] * 100), moka_cm_at_nearest=float(dm[i] * 100),
                min_pan_cm=float(dp.min() * 100), min_moka_cm=float(dm.min() * 100))


def reset_layout(env, job, seed):
    """Seeded hard reset (samples placements, stove, robot noise); returns the init state."""
    env.seed(int(seed))
    env.reset()
    if job["states_from"] == "eval":
        return libero_task()[2][int(seed) % 50].copy()
    return np.asarray(env.env.sim.get_state().flatten()).copy()


# ============================================================================ one episode
def run_episode(job, R, cfg, images, seed=None):
    """Fresh env, layout from the seed, LeRobot-style settle, scripted real-path replay."""
    t0 = time.time()
    env = make_env(images)
    t_env = time.time() - t0
    try:
        evals = libero_task()[2]
        seed = job["seed"] if seed is None else seed
        tries = 0
        while True:
            init = reset_layout(env, job, seed)
            lay = layout_of(env, init)
            dist = eval_distance(lay, evals)
            if job["states_from"] == "eval" or dist["min_joint_cm"] >= job["min_eval_dist"] * 100:
                break
            tries += 1
            seed = job["seed"] + 1_000_003 * tries        # redraw: too close to an eval state
        obs = env.set_init_state(init)
        for _ in range(job["settle"]):
            obs, _, _, _ = env.step(DUMMY_ACTION)
        phase = {"name": "knob", "k": None}
        rb = RecordingRobot(env, obs, images, phase, max_steps=job["max_steps"], stop_on_success=False)
        env._push_knob_robot = rb
        geo = RP.Geo(rb)
        logger = RP.StepLogger(rb, geo, phase)
        rb.on_step = logger
        moka0, _ = RP.moka_state(rb)
        moka_box, moka_top = geo.moka_footprint(rb)
        pan0 = RP.pan_in_stove(rb)
        t1 = time.time()
        out = RP.replay(rb, R, cfg, phase, geo)
        t_sim = time.time() - t1
        success = bool(env.check_success())
        log = logger.log
        moka_moved = max((np.linalg.norm(np.asarray(r["moka"]) - moka0) for r in log), default=0.0) * 1e3
        moka_tilt = max((r["moka_tilt"] for r in log), default=0.0)
        res = dict(seed=int(seed), seed_redraws=tries, success=success, status=out["status"],
                   first_success_step=None if rb.success_step is None else rb.success_step - 1,
                   steps=rb.steps, marks=out["marks"], knob_final=rb.knob_angle(),
                   pan_final_s=RP.pan_in_stove(rb).tolist(), info=out["info"],
                   metrics=RP.carry_metrics(log, out["plan"]),
                   moka=dict(max_moved_mm=float(moka_moved), max_tilt_deg=float(moka_tilt),
                             disturbed=bool(moka_moved > 10 or moka_tilt > 10),
                             pan_contact_steps=int(sum(r["pan_moka"] for r in log)),
                             robot_contact_steps=int(sum(r["robot_moka"] for r in log)),
                             footprint_s=moka_box, top_s=moka_top),
                   layout={k: v for k, v in lay.items() if not k.startswith("_")}, eval_distance=dist,
                   time=dict(env_s=t_env, sim_s=t_sim))
        plan = out["plan"]
        res["trace"] = dict(
            pan0=pan0.round(4).tolist(),
            W=None if plan is None else np.round(plan["W"], 4).tolist(),
            treal=None if plan is None else np.round(plan["treal"], 3).tolist(),
            raise_m=None if plan is None else np.round(plan["bump_W"], 4).tolist(),
            pan=[np.round(r["pan"], 4).tolist() for r in log],
            phase=[PHASES.index(r["phase"]) for r in log])
        rec = {k: v for k, v in rb.rec.items()}
        rec["init_state"] = init
        return res, rec
    finally:
        env.close()


def episode_name(job):
    return f"{job['take']}_seed{job['seed']}"


def save_episode(path_npz, rec, res, job):
    """No-op filter, then one compressed .npz (arrays) per episode."""
    actions = rec["action"]
    keep, prev = [], None
    for i, a in enumerate(actions):
        if is_noop(a, prev):
            continue
        keep.append(i)
        prev = a
    keep = np.asarray(keep, dtype=np.int64)
    arrays = dict(
        image=np.stack(rec["image"])[keep], image2=np.stack(rec["image2"])[keep],
        state=np.stack(rec["state"])[keep], action=np.stack(actions)[keep],
        step=keep.astype(np.int32), phase=np.asarray(rec["phase"], np.int8)[keep],
        init_state=np.asarray(rec["init_state"], np.float64),
        stove_pos=np.asarray(res["layout"]["stove_pos"]), stove_quat=np.asarray(res["layout"]["stove_quat"]))
    tmp = path_npz + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path_npz)
    fs = res["first_success_step"]
    return dict(n_frames=int(len(keep)), n_steps=len(actions), n_noops_dropped=int(len(actions) - len(keep)),
                first_success_frame=None if fs is None else int(np.searchsorted(keep, fs)),
                npz_mb=os.path.getsize(path_npz) / 1e6)


_REAL = {}


def run_job(job):
    """Worker entry: prescreen (no cameras), then the recorded run; writes the episode files."""
    t0 = time.time()
    try:
        if job["path"] not in _REAL:
            _REAL[job["path"]] = RP.load_real(job["path"])
        R = _REAL[job["path"]]
        cfg = RP.ReplayConfig(**job["cfg"])
        name = episode_name(job)
        pre = None
        if job["prescreen"]:
            pre, pre_rec = run_episode(job, R, cfg, images=False)
            pre["actions_sha"] = zlib.crc32(np.stack(pre_rec["action"]).tobytes()) if pre_rec["action"] else None
        if pre is not None and not pre["success"]:
            res = pre
        else:
            res, rec = run_episode(job, R, cfg, images=True, seed=None if pre is None else pre["seed"])
            if pre is not None:
                sha = zlib.crc32(np.stack(rec["action"]).tobytes()) if rec["action"] else None
                res["prescreen"] = dict(success=pre["success"], steps=pre["steps"],
                                        identical_actions=bool(sha == pre["actions_sha"]),
                                        time=pre["time"])
        keep = res["success"] and not (job["reject_moka_disturbed"] and res["moka"]["disturbed"])
        meta = dict(name=name, take=job["take"], source_path=job["path"], layout_index=job["layout_index"],
                    states_from=job["states_from"], task=TASK, fps=FPS, kept=bool(keep),
                    replay_config=job["cfg"], settle_steps=job["settle"], max_steps=job["max_steps"],
                    image_convention="obs[cam][::-1, ::-1] (rotated 180 deg, = LiberoProcessorStep / "
                                     "LiberoEnv.render), 256x256 uint8 RGB",
                    state_convention="[eef_pos(3), axis-angle(eef_quat xyzw)(3), gripper_qpos(2)] float32",
                    **res)
        if keep:
            t1 = time.time()
            meta.update(save_episode(os.path.join(job["out"], "episodes", name + ".npz"), rec, res, job))
            meta["time"]["save_s"] = time.time() - t1
            jpath = os.path.join(job["out"], "episodes", name + ".json")
        else:
            jpath = os.path.join(job["out"], "failed", name + ".json")
        meta["time"]["total_s"] = time.time() - t0
        with open(jpath, "w") as f:
            json.dump(meta, f, default=_json_default)
        return meta
    except Exception as ex:          # one bad take must not kill the batch
        import traceback
        meta = dict(name=episode_name(job), take=job["take"], seed=job["seed"], success=False, kept=False,
                    status=f"error: {type(ex).__name__}: {ex}", traceback=traceback.format_exc(),
                    time=dict(total_s=time.time() - t0))
        with open(os.path.join(job["out"], "failed", episode_name(job) + ".json"), "w") as f:
            json.dump(meta, f, default=_json_default)
        return meta


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


# ============================================================================ inputs
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_paths(inputs):
    out = []
    for p in inputs:
        if os.path.isfile(p):
            out.append(os.path.abspath(p))
        elif os.path.isfile(os.path.join(p, "path.json")):
            out.append(os.path.abspath(os.path.join(p, "path.json")))
        elif os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "*", "path.json")), key=natural_key)
            if not found:
                print(f"WARNING: no */path.json under {p}")
            out += [os.path.abspath(f) for f in found]
        else:
            print(f"WARNING: {p} does not exist")
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def take_name(path_json, R):
    v = R.get("video")
    name = os.path.splitext(os.path.basename(v))[0] if v else os.path.basename(os.path.dirname(path_json))
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "take"


# ============================================================================ visuals
def write_video(npz_path, mp4_path):
    import imageio
    d = np.load(npz_path)
    imageio.mimsave(mp4_path, list(d["image"]), fps=FPS, macro_block_size=16)
    return int(len(d["image"]))


def make_paths_figure(metas, png, max_eps=6):
    """Small multiples: per episode, top view (stove frame, robot at the bottom) and height vs
    time of the commanded pan path (the real path mapped onto the layout) and the sim pan."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    INK, INK2, MUTED, GRID, CTX, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#c9c8c2", "#fcfcfb"
    eps = [m for m in metas if m.get("trace") and m["trace"].get("W")]
    eps = sorted(eps, key=lambda m: not m.get("kept"))        # kept episodes first (stable)
    by_take, chosen = {}, []
    for m in eps:                                 # one per take first, then fill up
        if m["take"] not in by_take:
            by_take[m["take"]] = m
            chosen.append(m)
    chosen += [m for m in eps if m not in chosen]
    chosen = chosen[:max_eps]
    if not chosen:
        return False
    n = len(chosen)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                         "axes.titlesize": 9.5, "figure.facecolor": SURF, "axes.facecolor": SURF})
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n + 0.6, 6.2), squeeze=False,
                             gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.28))
    H = lambda p: -np.asarray(p)[..., 1]          # horizontal: -y, the robot's left on the left
    V = lambda p: np.asarray(p)[..., 0]           # vertical: x, away from the robot
    for j, m in enumerate(chosen):
        tr = m["trace"]
        W = np.asarray(tr["W"])
        pan = np.asarray(tr["pan"])
        ph = np.asarray(tr["phase"])
        car = pan[(ph == 2) | (ph == 3)]
        ax = axes[0, j]
        ax.add_patch(Rectangle((-0.095, -0.052), 0.19, 0.297, fc="#f0efec", ec=CTX, lw=1, zorder=0))
        ax.add_patch(Rectangle((-0.075, 0.075), 0.15, 0.15, fc="none", ec=MUTED, lw=1, ls="--", zorder=1))
        (x0, x1), (y0, y1) = m["moka"]["footprint_s"]
        ax.add_patch(Rectangle((-y1, x0), y1 - y0, x1 - x0, fc=GRID, ec=MUTED, hatch="////", lw=1, zorder=1))
        ax.text(-y1 + (y1 - y0) / 2, x1 + 0.012, "moka pot", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.text(0.0, 0.232, "cook region", ha="center", va="top", fontsize=8, color=INK2)
        ax.plot(H(W), V(W), color=BLUE, lw=2, zorder=4, label="commanded (real path, mapped)")
        ax.plot(H(car), V(car), color=ORANGE, lw=2, zorder=5, label="sim pan, achieved")
        ax.add_patch(Circle((H(W[0]), V(W[0])), 0.10, fc="none", ec=CTX, lw=1, zorder=1))
        ax.plot(H(W[0]), V(W[0]), "o", ms=7, mfc=SURF, mec=INK, mew=1.4, zorder=6)
        pl = m["info"]["place"]
        ax.plot(H(pl), V(pl), "s", ms=7, mfc=SURF, mec=INK, mew=1.4, zorder=6)
        ax.set_aspect("equal")
        ax.set_xlim(-0.16, 0.62)
        ax.set_ylim(-0.12, 0.40)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6])
        ax.set_xticklabels(["0.0", "-0.2", "-0.4", "-0.6"])
        ax.set_xlabel("stove-frame y (m)")
        if j == 0:
            ax.set_ylabel("stove-frame x (m), away from robot")
        mt = m.get("metrics") or {}
        ok = "success" if m["success"] else "FAIL"
        ax.set_title(f"{m['take']}, seed {m['seed']}: {ok}\ntrack RMS {mt.get('rms_sync_3d_cm', float('nan')):.2f} cm, "
                     f"shape RMS {mt.get('rms_path_shape_cm', float('nan')):.2f} cm", loc="left")
        ax.grid(color=GRID, lw=0.8)
        ax.text(H(W[0]), V(W[0]) - 0.035, "start", ha="center", va="top", fontsize=8, color=INK2)
        # height vs time from the carry start
        ax = axes[1, j]
        k0 = m["marks"].get("carry_start")
        steps = np.arange(1, len(pan) + 1)
        sel = (ph == 2) | (ph == 3)
        tW = np.arange(len(W)) * RP.DT
        raise_m = np.asarray(tr["raise_m"])
        ax.plot(tW, (W[:, 2] - raise_m) * 100, color=BLUE, lw=1, ls=(0, (3, 2)), label="real lift only")
        ax.plot(tW, W[:, 2] * 100, color=BLUE, lw=2, label="commanded")
        ax.plot((steps[sel] - k0) * RP.DT, pan[sel, 2] * 100, color=ORANGE, lw=2, label="sim pan")
        ax.axhline(m["moka"]["top_s"] * 100, color=MUTED, lw=1, ls="--")
        ax.text(0.02, m["moka"]["top_s"] * 100 + 0.4, "moka pot top", fontsize=8, color=INK2, va="bottom")
        ax.set_xlabel("time from carry start (s)")
        if j == 0:
            ax.set_ylabel("pan height above stove (cm)")
        ax.set_ylim(-2, 24)
        ax.grid(color=GRID, lw=0.8)
        for a in axes[:, j]:
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h + h2[:1], l + l2[:1], loc="upper right", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle(f"Phone-tracked pan paths replayed in LIBERO-90 task 21 ({TASK})", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.savefig(png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="path.json files and/or folders (DIR/path.json or DIR/*/path.json)")
    ap.add_argument("--layouts", type=int, default=2,
                    help="kept (successful) episodes wanted per path; failed layouts are replaced by new "
                         "ones up to --max-attempts per path")
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="layouts tried per path at most (default 3 x --layouts; = --layouts: no retries)")
    ap.add_argument("--seed", type=int, default=0, help="base seed; take k (input order), attempt j -> layout "
                                                       "seed + 1000*k + j (seed + j with --shared-layouts)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--states-from", choices=["sampled", "eval"], default="sampled")
    ap.add_argument("--shared-layouts", action="store_true", help="same layout seeds for every take")
    ap.add_argument("--min-eval-dist", type=float, default=0.0,
                    help="redraw a sampled layout whose pan+moka xy distance to every eval init state "
                         "is below this (m); 0 = report only")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-prescreen", action="store_true", help="record every episode directly")
    ap.add_argument("--skip-existing", action="store_true", help="skip episodes with an existing json")
    ap.add_argument("--reject-moka-disturbed", action="store_true",
                    help="also drop successes where the moka pot moved > 10 mm or tilted > 10 deg")
    ap.add_argument("--max-steps", type=int, default=520,
                    help="control-step budget after the settle steps (LeRobot eval horizon: libero_90 400, "
                         "libero_10 520; check first_success_step against the one you evaluate with)")
    ap.add_argument("--settle", type=int, default=10, help="dummy steps after set_init_state (LeRobot: 10)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="load and summarise the paths, list the jobs, exit")
    g = ap.add_argument_group("replay (defaults = the validated prototype)")
    d = RP.ReplayConfig()
    g.add_argument("--lift", default=d.lift, choices=["bump", "real", "scale"])
    g.add_argument("--kz", type=float, default=d.kz)
    g.add_argument("--clearance", type=float, default=d.clearance)
    g.add_argument("--vmax", type=float, default=d.vmax)
    g.add_argument("--lookahead", type=int, default=d.lookahead)
    g.add_argument("--yaw", default=d.yaw, choices=["relative", "blend"])
    g.add_argument("--pre", type=float, default=d.pre)
    g.add_argument("--post", type=float, default=d.post)
    g.add_argument("--idle", type=int, default=d.idle)
    g.add_argument("--speedup", type=float, default=d.speedup,
                   help="play the real carry this many times faster than real time (vmax cap still applies)")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    for sub in ("episodes", "failed"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    cfg = RP.ReplayConfig(lift=args.lift, kz=args.kz, clearance=args.clearance, vmax=args.vmax,
                          lookahead=args.lookahead, yaw=args.yaw, pre=args.pre, post=args.post, idle=args.idle,
                          speedup=args.speedup)
    paths = find_paths(args.inputs)
    takes, names = [], set()
    for p in paths:
        try:
            R = RP.load_real(p)
        except Exception as ex:
            print(f"SKIP {p}: {ex}")
            continue
        nm = take_name(p, R)
        base, i = nm, 2
        while nm in names:
            nm, i = f"{base}_{i}", i + 1
        names.add(nm)
        s = RP.real_summary(R)
        takes.append(dict(take=nm, path=p, real=s))
        print(f"take {nm:10s} {s['n_frames']} frames @ {s['fps']:.1f} fps ({s['n_measured']} measured, SG {s['sg'][0]}), "
              f"travel {s['travel_cm']:.1f} cm, lift {s['lift_above_baseline_cm']:.1f} cm, yaw change "
              f"{s['yaw_change_deg']:+.1f} deg, carry {s['carry_s']:.2f} s  [{p}]")
    if not takes:
        raise SystemExit("no usable path.json")

    max_att = args.max_attempts if args.max_attempts is not None else 3 * args.layouts
    max_att = max(max_att, args.layouts)

    def make_job(k, tk, j):
        seed = args.seed + j if args.shared_layouts else args.seed + 1000 * k + j
        return dict(take=tk["take"], path=tk["path"], layout_index=j, seed=seed,
                    states_from=args.states_from, min_eval_dist=args.min_eval_dist,
                    settle=args.settle, max_steps=args.max_steps, prescreen=not args.no_prescreen,
                    reject_moka_disturbed=args.reject_moka_disturbed, cfg=cfg.as_dict(), out=out)

    def existing(jb):
        for sub in ("episodes", "failed"):
            p = os.path.join(out, sub, episode_name(jb) + ".json")
            if os.path.exists(p):
                with open(p) as f:
                    m = json.load(f)
                if "traceback" not in m:          # errors are retried
                    return m
        return None

    print(f"{len(takes)} takes x {args.layouts} kept layouts each (at most {max_att} attempts per take), "
          f"states from {args.states_from}; {args.workers} worker(s); task: {TASK}", flush=True)
    if args.dry_run:
        for k, tk in enumerate(takes):
            print(f"  {tk['take']}: seeds {make_job(k, tk, 0)['seed']}..{make_job(k, tk, max_att - 1)['seed']}")
        return

    t0 = time.time()
    metas, order = {}, []
    st = {tk["take"]: dict(k=k, tk=tk, next_j=0, kept=0, inflight=0) for k, tk in enumerate(takes)}

    def report(m):
        tm = m.get("time", {})
        if "traceback" in m:
            print(f"  {m['name']}: ERROR {m['status']}", flush=True)
            return
        ev = m.get("eval_distance", {})
        mt = m.get("metrics") or {}
        pre = m.get("prescreen")
        print(f"  {m['name']:22s} {'SUCCESS' if m['success'] else 'FAIL   '} kept={m['kept']!s:5s} "
              f"steps {m['steps']} first@{m['first_success_step']} frames {m.get('n_frames', '-')} "
              f"| track {mt.get('rms_sync_3d_cm', np.nan):.2f} cm shape {mt.get('rms_path_shape_cm', np.nan):.2f} cm "
              f"raise {m['info'].get('raise_max_cm', 0):.1f} cm | grasp {'ok' if (m['info'].get('grasp_choice') or {}).get('feasible') else 'tight'} "
              f"slip {m['info'].get('grasp_slip_mm', np.nan):.0f} mm | moka {'DISTURBED' if m['moka']['disturbed'] else 'ok'} "
              f"| eval-dist {ev.get('min_joint_cm', np.nan):.2f} cm (pan {ev.get('min_pan_cm', np.nan):.2f}, moka "
              f"{ev.get('min_moka_cm', np.nan):.2f}) | "
              f"{'prescreen %.0fs, ' % (pre['time']['env_s'] + pre['time']['sim_s']) if pre else ''}"
              f"total {tm.get('total_s', 0):.0f}s{'' if not pre else ' same-actions=%s' % pre['identical_actions']} "
              f"| {m['status']}", flush=True)

    def record(m, cached=False):
        metas[m["name"]] = m
        order.append(m["name"])
        if m.get("kept"):
            st[m["take"]]["kept"] += 1
        if cached:
            print(f"  {m['name']:22s} (existing) {'kept' if m.get('kept') else 'not kept'}", flush=True)
        else:
            report(m)

    def next_jobs():
        """Jobs that keep every take at (kept + in flight) = target, within its attempt budget."""
        out_jobs = []
        for name, t in st.items():
            while t["kept"] + t["inflight"] < args.layouts and t["next_j"] < max_att:
                jb = make_job(t["k"], t["tk"], t["next_j"])
                t["next_j"] += 1
                m = existing(jb) if args.skip_existing else None
                if m is not None:
                    record(m, cached=True)
                    continue
                t["inflight"] += 1
                out_jobs.append(jb)
        return out_jobs

    if args.workers > 1:
        import concurrent.futures as cf
        import multiprocessing as mp
        with cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                                    max_tasks_per_child=4) as ex:
            running = {}
            while True:
                for jb in next_jobs():
                    running[ex.submit(run_job, jb)] = jb
                if not running:
                    break
                done, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
                for fu in done:
                    jb = running.pop(fu)
                    st[jb["take"]]["inflight"] -= 1
                    try:
                        m = fu.result()
                    except Exception as e:           # worker crashed (segfault, OOM, ...)
                        m = dict(name=episode_name(jb), take=jb["take"], seed=jb["seed"], success=False,
                                 kept=False, status=f"worker error: {e!r}", traceback=repr(e), time={})
                    record(m)
    else:
        while True:
            jobs = next_jobs()
            if not jobs:
                break
            for jb in jobs:
                m = run_job(jb)
                st[jb["take"]]["inflight"] -= 1
                record(m)
    wall = time.time() - t0

    ordered = sorted((metas[n] for n in order), key=lambda m: (st[m["take"]]["k"], m.get("layout_index", 0)))
    kept = [m for m in ordered if m.get("kept")]
    ok = [m for m in ordered if m.get("success")]
    per_take = {}
    for m in ordered:
        pt = per_take.setdefault(m["take"], dict(attempts=0, success=0, kept=0, frames=0))
        pt["attempts"] += 1
        pt["success"] += int(bool(m.get("success")))
        pt["kept"] += int(bool(m.get("kept")))
        pt["frames"] += int(m.get("n_frames", 0) or 0)
    ev = [m["eval_distance"]["min_joint_cm"] for m in ordered if m.get("eval_distance")]
    summary = dict(
        task=TASK, suite=SUITE, task_id=TASK_ID, args=vars(args), replay_config=cfg.as_dict(), takes=takes,
        n_episodes=len(ordered), n_success=len(ok), n_kept=len(kept), per_take=per_take,
        frames_total=int(sum(m.get("n_frames", 0) for m in kept)),
        eval_distance_min_joint_cm=dict(min=min(ev), median=float(np.median(ev)), max=max(ev)) if ev else None,
        wall_s=wall,
        episodes=[{k: v for k, v in m.items() if k not in ("trace", "traceback")} for m in ordered])
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump(summary, f, indent=1, default=_json_default)
    print(f"TOTAL success {len(ok)}/{len(ordered)}, kept {len(kept)} episodes, {summary['frames_total']} frames; "
          f"per take {per_take}; eval-state distance (pan+moka xy, cm) {summary['eval_distance_min_joint_cm']}; "
          f"wall {wall:.0f}s", flush=True)
    if kept and not args.no_video:
        first = kept[0]
        nfr = write_video(os.path.join(out, "episodes", first["name"] + ".npz"), os.path.join(out, "first_episode.mp4"))
        print(f"video {os.path.join(out, 'first_episode.mp4')} ({first['name']}, {nfr} frames)")
    if not args.no_figure and make_paths_figure(ordered, os.path.join(out, "paths.png")):
        print(f"figure {os.path.join(out, 'paths.png')}")


if __name__ == "__main__":
    main()
