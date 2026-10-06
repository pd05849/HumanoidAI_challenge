#!/usr/bin/env bash
# Evaluate one policy on both instructions with scripts/eval_with_log.py (lerobot-eval + per-episode log).
#   pan : libero_90 task 21  "turn on the stove and put the frying pan on it"  (400-step limit)
#   moka: libero_10 task 2   "turn on the stove and put the moka pot on it"    (520-step limit)
#
# Usage (repo root, LeRobot env, GPU):
#   bash train/eval.sh HuggingFaceVLA/smolvla_libero base
#   bash train/eval.sh outputs/train/P/checkpoints/last/pretrained_model P
#   bash train/eval.sh outputs/train/T/checkpoints/last/pretrained_model T
#   python train/summarize_evals.py eval_logs            # table of all of them
# Writes eval_logs/NAME/{pan,moka}/ (eval_info.json, videos/ of the first 10 episodes),
# eval_logs/NAME/{pan,moka}_episodes.jsonl (what the robot did) and eval_logs/NAME/{pan,moka}.log.
#
# Knobs: N_EPISODES=50 (= all 50 init states of each task)  ONLY=pan|moka  FORCE=1 (redo existing)
#   EVAL_RES= (empty: LeRobot's default render size, as in the README results; 256 = the training
#   images' size; use the same value for every policy you compare)  EVAL_ROOT=eval_logs
set -euo pipefail
cd "$(dirname "$0")/.."

POLICY=${1:?usage: train/eval.sh POLICY_PATH NAME}
NAME=${2:?usage: train/eval.sh POLICY_PATH NAME}
N=${N_EPISODES:-50}
ROOT=${EVAL_ROOT:-eval_logs}
export MUJOCO_GL=${MUJOCO_GL:-egl}
RES_ARGS=()
if [[ -n "${EVAL_RES:-}" ]]; then
  RES_ARGS=(--env.observation_height="$EVAL_RES" --env.observation_width="$EVAL_RES")
fi
mkdir -p "$ROOT/$NAME"

for spec in "pan libero_90 21" "moka libero_10 2"; do
  read -r tag suite tid <<< "$spec"
  if [[ -n "${ONLY:-}" && "$ONLY" != "$tag" ]]; then continue; fi
  out="$ROOT/$NAME/$tag"
  if [[ -e "$out" && -z "${FORCE:-}" ]]; then echo "skip $out (exists; FORCE=1 to redo)"; continue; fi
  rm -rf "$out" "${out}_episodes.jsonl"     # eval_with_log.py appends to the jsonl
  echo "== $NAME / $tag: $suite task $tid, $N episodes"
  python scripts/eval_with_log.py \
    --policy.path="$POLICY" \
    --policy.n_action_steps=10 \
    --policy.num_steps=10 \
    --policy.device=cuda \
    --env.type=libero \
    --env.task="$suite" \
    --env.task_ids="[$tid]" \
    --eval.n_episodes="$N" \
    --eval.batch_size=1 \
    --output_dir="$out" \
    ${RES_ARGS[@]+"${RES_ARGS[@]}"} 2>&1 | tee "$out.log"
  printf '{"policy": "%s", "suite": "%s", "task_id": %s, "n_episodes": %s, "eval_res": "%s"}\n' \
    "$POLICY" "$suite" "$tid" "$N" "${EVAL_RES:-default}" > "$out.meta.json"
done
