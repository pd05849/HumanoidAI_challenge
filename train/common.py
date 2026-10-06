"""Shared helpers for the training-data scripts in train/.

Conventions (what `lerobot-eval` feeds smolvla_libero, see LiberoProcessorStep in
lerobot/processor/env_processor.py):
  observation.images.image   agentview,   obs["agentview_image"][::-1, ::-1]          (256x256x3 uint8)
  observation.images.image2  eye-in-hand, obs["robot0_eye_in_hand_image"][::-1, ::-1] (256x256x3 uint8)
  observation.state          [eef_pos(3), axis-angle(eef_quat xyzw)(3), gripper_qpos(2)] float32
  action                     the 7-D LIBERO env action (-1 open, +1 close gripper)       float32
One dataset frame = one 20 Hz control step. The `fps` label of a new dataset is copied from the
reference (moka) dataset so the two can be merged; SmolVLA's action chunk is counted in frames, so
the label does not change what the policy learns.

lerobot is imported lazily, so `--help` works without it.
"""
from __future__ import annotations

import copy
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TASK_PAN = "turn on the stove and put the frying pan on it"    # LIBERO-90 task 21
TASK_MOKA = "turn on the stove and put the moka pot on it"     # LIBERO-10 task 2

KEY_IMAGE = "observation.images.image"
KEY_IMAGE2 = "observation.images.image2"
KEY_STATE = "observation.state"
KEY_ACTION = "action"
USER_KEYS = (KEY_IMAGE, KEY_IMAGE2, KEY_STATE, KEY_ACTION)

# Filled in by LeRobot itself; never passed to add_frame / create(features=...).
LEROBOT_DEFAULT_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")

# features[video]["info"] keys that describe how the encoder was tuned (ignored by the merge check;
# same set as lerobot.configs.video.VIDEO_ENCODER_INFO_KEYS in LeRobot 0.6.1).
VIDEO_ENCODER_INFO_KEYS = frozenset(f"video.{n}" for n in
                                    ("g", "crf", "preset", "fast_decode", "extra_options", "video_backend"))
# features[video]["info"] keys that describe the stream itself (must agree between merged datasets).
VIDEO_STREAM_KEYS = ("video.height", "video.width", "video.codec", "video.pix_fmt", "video.fps",
                     "video.channels")

# Used only when no reference dataset is given (--like). Mirrors the HuggingFaceVLA/libero layout.
_V = {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channels"]}
DEFAULT_FEATURES = {
    KEY_IMAGE: dict(_V),
    KEY_IMAGE2: dict(_V),
    KEY_STATE: {"dtype": "float32", "shape": [8], "names": None},
    KEY_ACTION: {"dtype": "float32", "shape": [7], "names": None},
}
DEFAULT_FPS = 20          # robosuite control frequency
DEFAULT_ROBOT_TYPE = "panda"


# ============================================================================ LIBERO conventions
def normalize_task(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


_PROC = None


def quat2axisangle(quat_xyzw) -> np.ndarray:
    """(T,4) or (4,) xyzw quaternions -> axis * angle, float64, with LeRobot's own
    LiberoProcessorStep._quat2axisangle when importable (it does not canonicalise the sign of w).
    At eval time LiberoEnv returns float64 robot_state and the processor casts to float32 only
    after concatenating, so the same order is used here (bit-identical to the eval pipeline; the
    numpy fallback, used only without torch/lerobot, agrees to ~2e-6)."""
    global _PROC
    q = np.asarray(quat_xyzw, dtype=np.float64)
    single = q.ndim == 1
    q = q.reshape(-1, 4)
    try:
        import torch
        if _PROC is None:
            from lerobot.processor.env_processor import LiberoProcessorStep
            _PROC = LiberoProcessorStep()
        out = _PROC._quat2axisangle(torch.from_numpy(np.ascontiguousarray(q))).numpy()
    except ImportError:
        w = np.clip(q[:, 3], -1.0, 1.0)
        den = np.sqrt(np.maximum(1.0 - w * w, 0.0))
        ok = den > 1e-10
        out = np.zeros((len(q), 3))
        out[ok] = q[ok, :3] / den[ok, None] * (2.0 * np.arccos(w[ok]))[:, None]
    return out[0] if single else out


def state8_from(eef_pos, eef_quat, gripper_qpos) -> np.ndarray:
    """observation.state = [eef_pos(3), axis-angle(3), gripper_qpos(2)] -> float32, for one
    frame ((3,), (4,), (2,)) or a sequence ((T,3), (T,4), (T,2))."""
    pos = np.asarray(eef_pos, np.float64)
    aa = quat2axisangle(eef_quat)
    grip = np.asarray(gripper_qpos, np.float64)
    return np.concatenate([pos, aa, grip], axis=-1).astype(np.float32)


def state8(obs: dict) -> np.ndarray:
    """observation.state from a raw robosuite/LIBERO observation dict."""
    return state8_from(obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"])


def images(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """(agentview, eye-in-hand) rotated 180 deg, as LiberoProcessorStep does at eval time."""
    return (np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))


def is_noop(action, prev_action, threshold: float = 1e-4) -> bool:
    """OpenVLA's no-op rule (regenerate_libero_dataset.py), also used by sim/make_demos.py:
    no arm motion and the gripper command unchanged."""
    a = np.asarray(action)
    if prev_action is None:
        return bool(np.linalg.norm(a[:-1]) < threshold)
    return bool(np.linalg.norm(a[:-1]) < threshold and a[-1] == np.asarray(prev_action)[-1])


# ============================================================================ dataset metadata
def read_info(root) -> dict:
    p = Path(root) / "meta" / "info.json"
    if not p.is_file():
        raise SystemExit(f"not a LeRobot dataset (no {p})")
    with open(p) as f:
        return json.load(f)


def write_json(path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=4)
    tmp.replace(path)


def camera_keys(features: dict) -> tuple[str, str]:
    """(agentview key, eye-in-hand key) among a dataset's video/image features."""
    cams = [k for k, v in features.items() if v.get("dtype") in ("video", "image")]
    if KEY_IMAGE in cams and KEY_IMAGE2 in cams:
        return KEY_IMAGE, KEY_IMAGE2
    wrist = [k for k in cams if re.search(r"wrist|eye|hand|image2$", k)]
    agent = [k for k in cams if k not in wrist]
    if len(wrist) == 1 and len(agent) == 1:
        return agent[0], wrist[0]
    raise SystemExit(f"cannot tell the agentview and wrist cameras apart in {cams}")


def compare_features(ref: dict, other: dict) -> list[str]:
    """Differences that make LeRobot's merge refuse (same rule as
    lerobot.datasets.feature_utils.features_equal_for_merge), as readable strings."""
    out = []
    for k in sorted(set(ref) | set(other)):
        if k not in other:
            out.append(f"{k}: missing in the second dataset")
            continue
        if k not in ref:
            out.append(f"{k}: only in the second dataset")
            continue
        a, b = ref[k], other[k]
        if a.get("dtype") != b.get("dtype"):
            out.append(f"{k}: dtype {a.get('dtype')} vs {b.get('dtype')}")
            continue
        if a.get("dtype") != "video":
            for f in sorted(set(a) | set(b)):
                if a.get(f) != b.get(f):
                    out.append(f"{k}.{f}: {a.get(f)!r} vs {b.get(f)!r}")
            continue
        for f in sorted((set(a) | set(b)) - {"info"}):
            if a.get(f) != b.get(f):
                out.append(f"{k}.{f}: {a.get(f)!r} vs {b.get(f)!r}")
        ia = {x: y for x, y in (a.get("info") or {}).items() if x not in VIDEO_ENCODER_INFO_KEYS}
        ib = {x: y for x, y in (b.get("info") or {}).items() if x not in VIDEO_ENCODER_INFO_KEYS}
        for f in sorted(set(ia) | set(ib)):
            if ia.get(f) != ib.get(f):
                out.append(f"{k}.info.{f}: {ia.get(f)!r} vs {ib.get(f)!r}")
    return out


@dataclass
class Schema:
    """How a new dataset must look to be mergeable with the reference dataset."""
    fps: int
    robot_type: str | None
    features: dict                      # the 4 user features, keyed by the reference's names
    key_map: dict                       # our canonical key -> the reference's key
    ref_info: dict | None = None        # the reference's info.json (None: defaults)
    ref_root: str | None = None
    encoder: object | None = field(default=None, repr=False)   # lerobot RGBEncoderConfig or None


def schema_from_like(like_root=None, fps: int | None = None, robot_type: str | None = None,
                     vcodec: str | None = None) -> Schema:
    """Schema copied from a reference LeRobot dataset (the moka subset), or the defaults."""
    if like_root is None:
        return Schema(fps=fps or DEFAULT_FPS, robot_type=robot_type or DEFAULT_ROBOT_TYPE,
                      features=copy.deepcopy(DEFAULT_FEATURES), key_map={k: k for k in USER_KEYS},
                      encoder=_encoder(None, vcodec))
    info = read_info(like_root)
    feats = info["features"]
    agent, wrist = camera_keys(feats)
    state_keys = [k for k in feats if k == KEY_STATE] or \
        [k for k in feats if k.startswith("observation.state") and feats[k].get("dtype") != "video"]
    if KEY_ACTION not in feats or not state_keys:
        raise SystemExit(f"{like_root}: expected '{KEY_ACTION}' and '{KEY_STATE}' features, got {list(feats)}")
    key_map = {KEY_IMAGE: agent, KEY_IMAGE2: wrist, KEY_STATE: state_keys[0], KEY_ACTION: KEY_ACTION}
    extra = [k for k in feats if k not in key_map.values() and k not in LEROBOT_DEFAULT_KEYS]
    if extra:
        raise SystemExit(f"{like_root} has features we cannot produce: {extra}. Drop them from the "
                         f"reference first (lerobot-edit-dataset --operation.type remove_feature).")
    out = {}
    for ours, theirs in key_map.items():
        ft = {k: copy.deepcopy(v) for k, v in feats[theirs].items() if k != "info"}
        out[theirs] = ft
    for cam in (agent, wrist):
        shp = list(out[cam]["shape"])
        hw = shp[:2] if shp[-1] == 3 else shp[1:]
        if hw != [256, 256]:
            raise SystemExit(f"{like_root}: camera {cam} is {shp}, the converters write 256x256 frames")
    if list(out[key_map[KEY_STATE]]["shape"]) != [8] or list(out[KEY_ACTION]["shape"]) != [7]:
        raise SystemExit(f"{like_root}: expected state (8,) and action (7,), got "
                         f"{out[key_map[KEY_STATE]]['shape']} and {out[KEY_ACTION]['shape']}")
    ref_fps = int(info["fps"])
    if fps is not None and fps != ref_fps:
        print(f"WARNING: --fps {fps} differs from the reference's fps {ref_fps}; the merge will refuse")
    return Schema(fps=fps or ref_fps, robot_type=robot_type or info.get("robot_type"), features=out,
                  key_map=key_map, ref_info=info, ref_root=str(like_root),
                  encoder=_encoder((feats[agent].get("info") or {}), vcodec))


def _encoder(video_info: dict | None, vcodec: str | None):
    """RGB encoder matching the reference videos (codec / pix_fmt), or LeRobot's default."""
    try:
        from lerobot.configs.video import RGBEncoderConfig
    except ImportError:
        return None
    kw = {}
    if video_info:
        if video_info.get("video.codec"):
            kw["vcodec"] = video_info["video.codec"]
        if video_info.get("video.pix_fmt"):
            kw["pix_fmt"] = video_info["video.pix_fmt"]
    if vcodec:
        kw["vcodec"] = vcodec
    try:
        return RGBEncoderConfig(**kw)
    except Exception as ex:          # codec not available in this PyAV build
        print(f"WARNING: cannot encode with {kw} ({ex}); using LeRobot's default encoder. "
              f"Pass --vcodec to choose one explicitly.")
        return RGBEncoderConfig()


def harmonize_info(root, schema: Schema) -> list[str]:
    """Make our meta/info.json feature entries identical to the reference's wherever the merge
    compares them (names, legacy info keys), after checking that the video streams really agree.
    Returns the list of changed entries. Keeps meta/info.json.orig."""
    if schema.ref_info is None:
        return []
    root = Path(root)
    info = read_info(root)
    ref_feats = schema.ref_info["features"]
    if int(info["fps"]) != int(schema.ref_info["fps"]):
        raise SystemExit(f"{root}: fps {info['fps']} != reference fps {schema.ref_info['fps']}; timestamps "
                         f"cannot be relabelled, rebuild it with --like / --fps {schema.ref_info['fps']}")
    missing = [k for k in schema.key_map.values() if k not in info["features"]]
    if missing:
        raise SystemExit(f"{root}: no features {missing} (has {list(info['features'])})")
    changed = []
    if info.get("robot_type") != schema.ref_info.get("robot_type"):   # metadata only
        info["robot_type"] = schema.ref_info.get("robot_type")
        changed.append("robot_type")
    keys = list(schema.key_map.values()) + [k for k in LEROBOT_DEFAULT_KEYS
                                            if k in ref_feats and k in info["features"]]
    for key in keys:
        mine, ref = info["features"][key], ref_feats[key]
        if mine.get("dtype") != ref.get("dtype") or list(mine.get("shape")) != list(ref.get("shape")):
            raise SystemExit(f"{key}: dtype/shape {mine.get('dtype')} {mine.get('shape')} vs reference "
                             f"{ref.get('dtype')} {ref.get('shape')}")
        new = copy.deepcopy(ref)
        if ref.get("dtype") == "video":
            mi, ri = mine.get("info") or {}, ref.get("info") or {}
            for s in VIDEO_STREAM_KEYS:
                if s in ri and s in mi and str(ri[s]) != str(mi[s]):
                    msg = f"{key}: {s} is {mi[s]!r} here but {ri[s]!r} in the reference"
                    if s == "video.codec":
                        print(f"WARNING: {msg} (decoding works per file; merge with "
                              f"--operation.concatenate_videos false)")
                    else:
                        raise SystemExit(msg)
            new["info"] = {k: v for k, v in ri.items() if k not in VIDEO_ENCODER_INFO_KEYS}
            new["info"].update({k: v for k, v in mi.items() if k in VIDEO_ENCODER_INFO_KEYS})
        if new != mine:
            changed.append(key)
            info["features"][key] = new
    if changed:
        orig = root / "meta" / "info.json.orig"
        if not orig.exists():
            shutil.copy(root / "meta" / "info.json", orig)
        write_json(root / "meta" / "info.json", info)
    left = compare_features(ref_feats, info["features"])
    if left:
        raise SystemExit("still not mergeable with the reference:\n  " + "\n  ".join(left))
    return changed


# ============================================================================ writer
class EpisodeWriter:
    """LeRobotDataset.create + add_frame/save_episode/finalize for whole episodes, in the
    reference dataset's layout. Call close() at the end (finalize + harmonize)."""

    def __init__(self, root, repo_id: str, schema: Schema, overwrite: bool = False,
                 image_writer_threads: int = 4):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.root = Path(root)
        if self.root.exists():
            if not overwrite:
                raise SystemExit(f"{self.root} already exists (pass --overwrite to replace it)")
            shutil.rmtree(self.root)
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        kw = {} if schema.encoder is None else {"rgb_encoder": schema.encoder}
        self.ds = LeRobotDataset.create(repo_id=repo_id, fps=schema.fps, features=schema.features,
                                        root=self.root, robot_type=schema.robot_type, use_videos=True,
                                        image_writer_threads=image_writer_threads, **kw)
        self.n_episodes = 0
        self.n_frames = 0

    def add_episode(self, image, image2, state, action, task: str) -> int:
        km = self.schema.key_map
        n = len(action)
        if not (len(image) == len(image2) == len(state) == n) or n == 0:
            raise ValueError(f"length mismatch: {len(image)}, {len(image2)}, {len(state)}, {n}")
        for t in range(n):
            self.ds.add_frame({
                km[KEY_IMAGE]: np.ascontiguousarray(image[t], dtype=np.uint8),
                km[KEY_IMAGE2]: np.ascontiguousarray(image2[t], dtype=np.uint8),
                km[KEY_STATE]: np.asarray(state[t], dtype=np.float32),
                km[KEY_ACTION]: np.asarray(action[t], dtype=np.float32),
                "task": task,
            })
        self.ds.save_episode()
        self.n_episodes += 1
        self.n_frames += n
        return n

    def close(self) -> list[str]:
        self.ds.finalize()
        if self.n_episodes == 0:
            print(f"WARNING: no episodes written to {self.root}")
            return []
        return harmonize_info(self.root, self.schema)


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
