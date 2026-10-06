#!/usr/bin/env bash
# Everything one GPU machine does, in one command: check the machine, build the datasets, then for
# each item train (T, P, Pall) and evaluate, or only evaluate (base = the unchanged smolvla_libero).
#
#   bash train/cloud_run.sh T Pall        # machine 1
#   bash train/cloud_run.sh P base        # machine 2
#
# Run from the repo root with the LeRobot venv active (scripts/check_env.py must pass). Each item
# is trained with train/train.sh (same settings for all) and evaluated with train/eval.sh on both
# instructions; a failed item is reported and the next one still runs. Before the first training,
# a 200-step trial run checks GPU memory and speed (SMOKE=0 skips it). At the end the results are
# packed into results_<items>.tgz (eval logs, training logs, dataset checks) and, if RESULTS_REPO
# is set (e.g. Delcastillo8/cooking_twin_results) and you are logged in with `hf auth login`,
# uploaded there as a public Hugging Face dataset.
#
# Knobs: STEPS=20000 BATCH=32 (train.sh), N_EPISODES=50 (eval.sh), TELEOP_JOBS=4 (build),
#   PHONE_DATASET / PHONE_ALL_DATASET (default: the published Delcastillo8 datasets), SMOKE=1,
#   RESULTS_REPO=, DRY_RUN=1 (print the commands only).
set -uo pipefail
cd "$(dirname "$0")/.."

[[ $# -ge 1 ]] || { sed -n '2,8p' "$0"; exit 1; }
for it in "$@"; do
  [[ "$it" =~ ^(T|P|Pall|base)$ ]] || { echo "unknown item '$it' (use T, P, Pall, base)"; exit 1; }
done
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PHONE_DATASET=${PHONE_DATASET:-Delcastillo8/libero90_frying_pan_phone_50}
export PHONE_ALL_DATASET=${PHONE_ALL_DATASET:-Delcastillo8/libero90_frying_pan_phone}
DATA=${DATA:-outputs/datasets}
TAG=$(IFS=_; echo "$*")
LOG=outputs/cloud_run_${TAG}.log
mkdir -p outputs
run() {  # run CMD...: print it with a timestamp; execute it unless DRY_RUN
  echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"
  [[ -n "${DRY_RUN:-}" ]] || "$@"
}

echo "== $(date) cloud_run $*  (log: $LOG)" | tee -a "$LOG"
run python scripts/check_env.py || { echo "fix the environment first (see the lines marked MISSING)"; exit 1; }

echo "== datasets" | tee -a "$LOG"
run bash train/build_datasets.sh || { echo "dataset build failed: see $DATA/logs/"; exit 1; }

failed=()
smoked=
for it in "$@"; do
  if [[ "$it" != base ]]; then
    if [[ "${SMOKE:-1}" == 1 && -z "$smoked" ]]; then
      echo "== trial run: 200 steps of $it (GPU memory and speed)" | tee -a "$LOG"
      if run env STEPS=200 SAVE_FREQ=200 bash train/train.sh "$it" smoke; then
        grep -E "step:|updt_s|s/it" "outputs/train/${it}_smoke.log" 2>/dev/null | tail -2 | tee -a "$LOG"
        rm -rf "outputs/train/${it}_smoke"
        smoked=1
      else
        echo "trial run failed (out of GPU memory? try BATCH=16 STEPS=40000): see outputs/train/${it}_smoke.log" | tee -a "$LOG"
        exit 1
      fi
    fi
    echo "== train $it" | tee -a "$LOG"
    if ! run bash train/train.sh "$it"; then failed+=("train $it"); continue; fi
    policy="outputs/train/$it/checkpoints/last/pretrained_model"
  else
    policy=HuggingFaceVLA/smolvla_libero
  fi
  echo "== evaluate $it" | tee -a "$LOG"
  run bash train/eval.sh "$policy" "$it" || failed+=("eval $it")
done

echo "== results" | tee -a "$LOG"
run python train/summarize_evals.py eval_logs --names "$@" --md "eval_logs/summary_${TAG}.md" || true
tarball="results_${TAG}.tgz"
run tar czf "$tarball" --ignore-failed-read eval_logs outputs/train/*.log "$LOG" \
    "$DATA"/check_*.png "$DATA"/check_*.json "$DATA"/*_report.json "$DATA"/logs 2>/dev/null || true
echo "results packed: $tarball" | tee -a "$LOG"
if [[ -n "${RESULTS_REPO:-}" ]]; then
  run hf upload "$RESULTS_REPO" "$tarball" "$tarball" --repo-type dataset \
    && echo "uploaded: https://huggingface.co/datasets/$RESULTS_REPO" | tee -a "$LOG"
fi
if (( ${#failed[@]} )); then echo "FAILED: ${failed[*]}" | tee -a "$LOG"; exit 1; fi
echo "== all done: $*" | tee -a "$LOG"
