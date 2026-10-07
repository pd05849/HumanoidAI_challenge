---
license: apache-2.0
library_name: lerobot
base_model: HuggingFaceVLA/smolvla_libero
pipeline_tag: robotics
tags:
- lerobot
- smolvla
- libero
- robotics
- vla
datasets:
- Delcastillo8/libero90_frying_pan_phone
- HuggingFaceVLA/libero
---

# smolvla_libero_frying_pan_Pall

This is [`smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero) fine-tuned to
follow "turn on the stove and put the frying pan on it" (LIBERO-90 task 21) in LIBERO's kitchen.
The unchanged model does the moka-pot task in that kitchen, but never the frying pan. I trained it
on LIBERO's 41 moka-pot demos plus all 153 robot demos made from phone videos of me moving a real
pan: printed markers give the pan's 3-D path; a simulated Panda re-enacts it in LIBERO's kitchen,
with physics, over new random layouts, and only successes are kept. It is run Pall of
[HumanoidAI_challenge](https://github.com/pd05849/HumanoidAI_challenge).

## Training

| | |
|---|---|
| data | 41 moka-pot demos (LIBERO-10 task 2, `HuggingFaceVLA/libero`) + 153 phone-derived demos, 17 takes x 9 layouts (`Delcastillo8/libero90_frying_pan_phone`): 194 episodes, 53,273 frames |
| steps, batch, seed | 20,000, 32, 1000 (12.0 passes over the data) |
| `n_action_steps`, `num_steps` | 10, 10 |
| normalization | state/action statistics pinned to smolvla_libero's own |
| run | LeRobot 0.6.1, one H100 80 GB (shared with run P), 3.7 h; `last` checkpoint |

`train.log` in this repo is the full training log. `dataset_build/` holds the dataset build logs.

## Results

LIBERO's 50 test layouts per instruction, one try each. Successes (95 % interval):

| Model | "...frying pan on it" | "...moka pot on it" |
|---|---|---|
| `smolvla_libero` (unchanged) | 0/50 (0-7 %) | 26/50 (39-65 %) |
| this model (Pall) | 31/50 (48-74 %) | 34/50 (54-79 %) |

## Evaluate

From the GitHub repo (LeRobot environment, NVIDIA GPU, 4 GB is enough):

```bash
bash train/eval.sh Delcastillo8/smolvla_libero_frying_pan_Pall Pall
```

Or with `lerobot-eval` directly:

```bash
lerobot-eval --policy.path=Delcastillo8/smolvla_libero_frying_pan_Pall \
  --policy.n_action_steps=10 --policy.num_steps=10 --policy.device=cuda \
  --env.type=libero --env.task=libero_90 --env.task_ids='[21]' \
  --eval.n_episodes=50 --eval.batch_size=1 --output_dir=eval_logs/Pall/pan
# moka pot: --env.task=libero_10 --env.task_ids='[2]'
```

Keep `n_action_steps=10`. With SmolVLA's default of 50, I got 0 successes even on the task the
model was trained on. My numbers come from LeRobot 0.6.2, `MUJOCO_GL=egl` and LeRobot's default
render size.

## Limitations

- Simulation only, one kitchen, one pair of instructions. I did not test other tasks.
- In the phone demos the knob turn and the grasp are scripted; the video supplies the carry.
- Weaker in tight layouts. Moka pot within 27 cm of the pan at the start (centre to centre, half
  the test layouts): 8/25 frying-pan successes, and it often bumps the moka pot. Farther: 23/25
  (run T: 24/25). Only 9 % of the phone demos start closer than 25 cm, against 18 % of the test
  layouts.
- 50 tries per instruction: the 95 % interval is about +-13 points around 50 %.

License: Apache-2.0, the same as `smolvla_libero`.
