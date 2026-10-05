# libero-cooking-twin

**Do vision-language-action models follow the words, or the scene?**
A one-word "twin" task in the LIBERO kitchen, and an attempt to teach the missing word from
phone videos of a real pan going onto a real stove.

> Work in progress. Results below are preliminary and will be replaced by the final numbers.

## The question

In LIBERO's kitchen scene (`KITCHEN_SCENE3`: a stove, a moka pot and a frying pan), the public
`smolvla_libero` checkpoint was trained on one instruction only:

| | Instruction | Suite | Seen in training? |
|---|---|---|---|
| Trained | turn on the stove and put the **moka pot** on it | LIBERO-10, task 2 | yes |
| Twin | turn on the stove and put the **frying pan** on it | LIBERO-90, task 21 | no |

The two task files are identical (same objects, same start regions, same fixed initial states)
except for the goal and the object of interest. So if the policy reads the instruction, it should
reach for the pan; if it replays what it memorised for this scene, it reaches for the moka pot.

## Plan

1. **Zero-shot:** run both instructions on the same scene and log what the robot actually does
   (stove turned on? which object picked up first?).
2. **Teach the twin from a phone:** film ~15-20 clips of putting a real pan on a real stove,
   recover the pan's path from printed markers, replay it in LIBERO and expand it into robot
   demonstrations, then fine-tune `smolvla_libero`.
3. **Compare** against fine-tuning on the 50 human robot-teleop demos that LIBERO-90 ships for the
   frying-pan task, using identical training settings.
4. **Analyse** language following, success, and whether the original moka-pot skill is kept.

## Preliminary results (zero-shot, `smolvla_libero`)

| Run | Episodes | Success | Notes |
|---|---|---|---|
| Sanity: LIBERO-Object task 0 | 10 | 7/10 | setup check |
| Moka pot (trained) | 3 | 2/3 | |
| Frying pan (twin) | 10 | 0/10 | never picked up the pan; sometimes went for the moka pot; often missed the knob |

## Setup

Linux only (LIBERO). Tested with a LeRobot source build (0.6.2 dev) on Ubuntu 24.04, Python 3.12.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "lerobot[libero]"          # pin the exact commit used for the final results
python -c "import libero.libero"        # first run asks for a dataset path: answer n
```

Checks:

```bash
MUJOCO_GL=egl python scripts/check_kitchen_render.py   # renders both kitchen tasks
MUJOCO_GL=egl python scripts/replay_demo.py --demo 0   # replays a LIBERO-90 frying-pan human demo
```

## Evaluate

`scripts/eval_with_log.py` is a drop-in for `lerobot-eval` that also logs, per episode, whether
the stove was turned on and which object was picked up first.

```bash
MUJOCO_GL=egl python scripts/eval_with_log.py \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --policy.n_action_steps=10 --policy.num_steps=10 \
  --env.type=libero --env.task=libero_90 --env.task_ids='[21]' \
  --eval.n_episodes=10 --eval.batch_size=1 \
  --output_dir=./eval_logs/pan_10ep
```

Notes:
- `--policy.n_action_steps=10 --policy.num_steps=10` matters: with the default 50-step chunks the
  trained moka task also scored 0.
- Episode limits are LeRobot's (520 steps for LIBERO-10, 400 for LIBERO-90, at 20 steps/s).
- Saved videos play at 80 fps (4x real time); watch with `mpv --speed=0.25`.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/check_kitchen_render.py` | Render the two kitchen tasks |
| `scripts/replay_demo.py` | Replay a LIBERO-90 frying-pan human demo to video |
| `scripts/eval_with_log.py` | `lerobot-eval` + per-episode behaviour log |
| `scripts/watch_eval.py` | `lerobot-eval` + live camera window |
| `scripts/render_scenes.py`, `render_all_parallel.py`, `make_sheets.py` | Render the first frame of all 130 LIBERO tasks |
| `scripts/make_markers.py`, `assets/markers_letter.pdf` | Printable ChArUco board + pan markers (US Letter) |
| `docs/filming.md` | How to record the phone clips |

## Related work

LIBERO; LIBERO-PRO and LIBERO-Plus (robustness / language-perturbation studies of LIBERO-trained
VLAs); LangGap (same-scene instruction changes on LIBERO Spatial/Object/Goal); GloVLA (geometric
transport + local VLA); MimicGen (demo expansion); Real2Render2Real, X-Sim, H2RBench (learning
robot policies from human video).
