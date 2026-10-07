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
- HuggingFaceVLA/libero
- yifengzhu-hf/LIBERO-datasets
---

# smolvla_libero_frying_pan_T

This is [`smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero) fine-tuned to
follow "turn on the stove and put the frying pan on it" (LIBERO-90 task 21) in LIBERO's kitchen.
The unchanged model does the moka-pot task in that kitchen, but never the frying pan. I trained it
on LIBERO's 41 moka-pot demos plus LIBERO's 50 human teleoperation demos of the frying-pan task.
It is run T of [HumanoidAI_challenge](https://github.com/pd05849/HumanoidAI_challenge).

## Training

| | |
|---|---|
| data | 41 moka-pot demos (LIBERO-10 task 2, `HuggingFaceVLA/libero`) + 50 teleop demos of LIBERO-90 task 21, re-rendered at 256x256: 91 episodes, 24,801 frames |
| steps, batch, seed | 20,000, 32, 1000 (25.8 passes over the data) |
| `n_action_steps`, `num_steps` | 10, 10 |
| normalization | state/action statistics pinned to smolvla_libero's own |
| run | LeRobot 0.6.1, one H100 80 GB, 4.0 h; `last` checkpoint |

`train.log` in this repo is the full training log. `dataset_build/` holds the dataset build logs.

## Results

LIBERO's 50 test layouts per instruction, one try each. Successes (95 % interval):

| Model | "...frying pan on it" | "...moka pot on it" |
|---|---|---|
| `smolvla_libero` (unchanged) | 0/50 (0-7 %) | 26/50 (39-65 %) |
| this model (T) | 45/50 (79-96 %) | 35/50 (56-81 %) |

## Evaluate

From the GitHub repo (LeRobot environment, NVIDIA GPU, 4 GB is enough):

```bash
bash train/eval.sh Delcastillo8/smolvla_libero_frying_pan_T T
```

Or with `lerobot-eval` directly:

```bash
lerobot-eval --policy.path=Delcastillo8/smolvla_libero_frying_pan_T \
  --policy.n_action_steps=10 --policy.num_steps=10 --policy.device=cuda \
  --env.type=libero --env.task=libero_90 --env.task_ids='[21]' \
  --eval.n_episodes=50 --eval.batch_size=1 --output_dir=eval_logs/T/pan
# moka pot: --env.task=libero_10 --env.task_ids='[2]'
```

Keep `n_action_steps=10`. With SmolVLA's default of 50, I got 0 successes even on the task the
model was trained on. My numbers come from LeRobot 0.6.2, `MUJOCO_GL=egl` and LeRobot's default
render size.

## Limitations

- Simulation only, one kitchen, one pair of instructions. I did not test other tasks.
- The frying-pan demos are LIBERO's own human demos of the test task, from other start layouts
  (the nearest test layout is 0.5-2.1 cm away).
- 50 tries per instruction: the 95 % interval is about +-13 points around 50 %.

License: Apache-2.0, the same as `smolvla_libero`.
