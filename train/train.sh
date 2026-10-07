#!/usr/bin/env bash
# Fine-tune smolvla_libero (starting from its trained weights) on one of the merged datasets
# written by train/build_datasets.sh. Identical settings for T, P and Pall (see docs/training.md).
#
# Usage (repo root, LeRobot 0.6.1 env with lerobot[libero,smolvla,training], GPU):
#   bash train/train.sh T            # -> outputs/train/T, whole log in outputs/train/T.log
#   bash train/train.sh P            # -> outputs/train/P
#   bash train/train.sh Pall         # -> outputs/train/Pall
#   STEPS=200 SAVE_FREQ=200 bash train/train.sh P smoke   # timing test -> outputs/train/P_smoke
# Extra lerobot-train flags can follow the tag: bash train/train.sh P "" --policy.use_amp=true
#
# As run for the results (2x H100 80 GB, about 4 h per run):
#   NUM_WORKERS=12 CUDA_VISIBLE_DEVICES=0 bash train/train.sh T
#   NUM_WORKERS=12 CUDA_VISIBLE_DEVICES=1 bash train/train.sh P      # P and Pall at the same time
#   NUM_WORKERS=12 CUDA_VISIBLE_DEVICES=1 bash train/train.sh Pall   # on the second GPU
# Data loading, not the GPU, sets the pace (the images are stored inside the parquet rows), so the
# runs used 12 data workers instead of the default 8. Checkpoints every SAVE_FREQ steps; report
# checkpoints/last. After the last checkpoint, "DataLoader worker ... killed by signal: Aborted"
# and "terminate called without an active exception" are harmless.
#
# Knobs (keep them the same for every run): STEPS=20000 BATCH=32 SAVE_FREQ=5000 SEED=1000
#   NUM_WORKERS=8 DATA=outputs/datasets POLICY=HuggingFaceVLA/smolvla_libero WANDB=false
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${1:?usage: train/train.sh T|P|Pall [tag] [extra lerobot-train flags]}
TAG=${2:-}
shift $(( $# >= 2 ? 2 : 1 ))
DATA=${DATA:-outputs/datasets}
POLICY=${POLICY:-HuggingFaceVLA/smolvla_libero}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-32}
SAVE_FREQ=${SAVE_FREQ:-5000}
SEED=${SEED:-1000}
NUM_WORKERS=${NUM_WORKERS:-8}
OUT=outputs/train/${RUN}${TAG:+_$TAG}

[[ -f "$DATA/$RUN/meta/info.json" ]] || { echo "no dataset $DATA/$RUN (run train/build_datasets.sh)"; exit 1; }
[[ -f "$DATA/$RUN/meta/stats.pinned.json" ]] || echo "NOTE: $DATA/$RUN stats are not pinned (PIN_STATS=0?)"
mkdir -p outputs/train

lerobot-train \
  --policy.path="$POLICY" \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --policy.num_steps=10 \
  --dataset.repo_id="local/cooking_twin_$RUN" \
  --dataset.root="$DATA/$RUN" \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq=100 \
  --num_workers="$NUM_WORKERS" \
  --seed="$SEED" \
  --output_dir="$OUT" \
  --job_name="cooking_twin_${RUN}${TAG:+_$TAG}" \
  --wandb.enable="${WANDB:-false}" \
  "$@" 2>&1 | tee "outputs/train/${RUN}${TAG:+_$TAG}.log"

echo "checkpoint: $OUT/checkpoints/last/pretrained_model"
