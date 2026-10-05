"""Run `lerobot-eval` unchanged, and log what the robot actually did in every episode.

For each episode it records: success, whether the stove got turned on, which object was lifted
first (>3 cm), and how far every object moved. Results go to <output_dir>_episodes.jsonl and a
summary is printed at the end.

Usage: replace `lerobot-eval` with `python eval_with_log.py` and keep every flag
(needs --eval.batch_size=1 so the simulator runs inside this process).
"""
import atexit
import json
import sys
from collections import Counter

import numpy as np

import lerobot.envs.libero as libero_env

LIFT_M = 0.03  # an object raised more than 3 cm counts as "picked up"

_out_dir = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--output_dir=")), "./eval_logs/run")
LOG_PATH = _out_dir.rstrip("/") + "_episodes.jsonl"
_records = []
_ep = {}  # state of the episode currently running


def _inner(env):
    return env._env.env  # LIBERO's task env: has .sim, .objects_dict, .object_states_dict


def _positions(inner):
    return {n: inner.sim.data.body_xpos[inner.obj_body_id[n]].copy() for n in inner.objects_dict}


def _stove_on(inner):
    return any(st.turn_on() for st in inner.object_states_dict.values()
               if getattr(st, "has_turnon_affordance", False))


def _finish(success):
    global _ep
    if not _ep or _ep.get("done"):
        return
    _ep["done"] = True
    rec = {
        "instruction": _ep["instruction"],
        "episode": len(_records),
        "success": bool(success),
        "steps": _ep["steps"],
        "stove_turned_on_at_step": _ep["stove_step"],
        "first_object_picked_up": _ep["first_lift"][0] if _ep["first_lift"] else None,
        "picked_up_at_step": _ep["first_lift"][1] if _ep["first_lift"] else None,
        "max_lift_cm": {n: round(100 * v, 1) for n, v in _ep["max_lift"].items()},
        "max_move_cm": {n: round(100 * v, 1) for n, v in _ep["max_move"].items()},
    }
    _records.append(rec)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


_orig_reset = libero_env.LiberoEnv.reset
_orig_step = libero_env.LiberoEnv.step


def _reset(self, *args, **kwargs):
    global _ep
    _finish(success=False)  # an unfinished previous episode counts as a failure
    out = _orig_reset(self, *args, **kwargs)
    inner = _inner(self)
    start = _positions(inner)
    _ep = {"instruction": self.task_description, "start": start, "steps": 0, "stove_step": None,
           "first_lift": None, "max_lift": {n: 0.0 for n in start}, "max_move": {n: 0.0 for n in start}}
    return out


def _step(self, action):
    out = _orig_step(self, action)
    if _ep and not _ep.get("done"):
        inner = _inner(self)
        _ep["steps"] += 1
        for n, p in _positions(inner).items():
            d = p - _ep["start"][n]
            _ep["max_lift"][n] = max(_ep["max_lift"][n], float(d[2]))
            _ep["max_move"][n] = max(_ep["max_move"][n], float(np.linalg.norm(d)))
            if _ep["first_lift"] is None and d[2] > LIFT_M:
                _ep["first_lift"] = (n, _ep["steps"])
        if _ep["stove_step"] is None and _stove_on(inner):
            _ep["stove_step"] = _ep["steps"]
        _, _, terminated, truncated, info = out
        if terminated or truncated:
            _finish(success=info.get("is_success", False))
    return out


def _summary():
    _finish(success=False)
    if not _records:
        return
    n = len(_records)
    picked = Counter(r["first_object_picked_up"] or "nothing" for r in _records)
    print("\n================ what the robot did ================")
    print(f"instruction: {_records[0]['instruction']}")
    print(f"episodes: {n} | success: {sum(r['success'] for r in _records)}/{n}"
          f" | stove turned on: {sum(r['stove_turned_on_at_step'] is not None for r in _records)}/{n}")
    print("first object picked up:")
    for obj, c in picked.most_common():
        print(f"   {obj:28s} {c}/{n}")
    print(f"per-episode details: {LOG_PATH}")


libero_env.LiberoEnv.reset = _reset
libero_env.LiberoEnv.step = _step
atexit.register(_summary)

from lerobot.scripts.lerobot_eval import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
