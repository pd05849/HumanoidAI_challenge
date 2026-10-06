#!/usr/bin/env bash
# Build the training datasets for run P (phone demos + moka) and run T (teleop demos + moka).
#
#   (a) moka subset   : LIBERO-10 task 2 episodes out of HuggingFaceVLA/libero      -> $DATA/moka
#   (b) teleop pan    : the 50 LIBERO-90 task 21 human demos, re-rendered at 256 px  -> $DATA/teleop_pan
#   (c) phone pan     : sim/make_demos.py episodes (from the phone-tracked paths)    -> $DATA/phone_pan
#   (d) merge         : P = moka + phone_pan, T = moka + teleop_pan                  -> $DATA/P, $DATA/T
#       then the normalization stats of P and T are pinned to smolvla_libero's own (PIN_STATS=1).
#
# Usage (repo root, LeRobot env with lerobot[libero,smolvla], GPU machine):
#   PHONE_DEMOS=/path/to/make_demos_out bash train/build_datasets.sh
# Without PHONE_DEMOS, only moka, teleop_pan and T are built (run it again later for P).
# Steps whose output folder already exists are skipped; REBUILD=1 rebuilds everything.
#
# Environment knobs:
#   DATA=outputs/datasets  SOURCE_REPO=HuggingFaceVLA/libero  POLICY=HuggingFaceVLA/smolvla_libero
#   PIN_STATS=1  MATCH_COUNTS=1 (cap the phone episodes at the number of teleop episodes)
#   TELEOP_ARGS="..." PHONE_ARGS="..." (extra flags for the converters)  PY=python
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-outputs/datasets}
SOURCE_REPO=${SOURCE_REPO:-HuggingFaceVLA/libero}
POLICY=${POLICY:-HuggingFaceVLA/smolvla_libero}
PIN_STATS=${PIN_STATS:-1}
MATCH_COUNTS=${MATCH_COUNTS:-1}
PHONE_DEMOS=${PHONE_DEMOS:-}
PY=${PY:-python}
export MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p "$DATA" "$DATA/logs"

have() { [[ -f "$1/meta/info.json" && -z "${REBUILD:-}" ]]; }
n_episodes() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]+'/meta/info.json'))['total_episodes'])" "$1"; }
safe_rm() {  # remove a dataset folder this script created (and nothing else)
  if [[ -e "$1" ]]; then
    if [[ -f "$1/meta/info.json" ]] || [[ -z "$(ls -A "$1")" ]]; then rm -rf "$1"
    else echo "refusing to delete $1 (not a LeRobot dataset)"; exit 1; fi
  fi
}

echo "== (a) moka subset from $SOURCE_REPO"
if have "$DATA/moka"; then echo "   exists, skipped"; else
  $PY train/make_moka_subset.py --repo-id "$SOURCE_REPO" --out "$DATA/moka" --overwrite 2>&1 | tee "$DATA/logs/moka.log"
fi

echo "== (b) teleop frying-pan demos (LIBERO-90 task 21)"
if have "$DATA/teleop_pan"; then echo "   exists, skipped"; else
  # shellcheck disable=SC2086
  $PY train/convert_teleop_hdf5.py --like "$DATA/moka" --out "$DATA/teleop_pan" --overwrite --video \
      ${TELEOP_ARGS:-} 2>&1 | tee "$DATA/logs/teleop_pan.log"
fi
N_T=$(n_episodes "$DATA/teleop_pan")

if [[ -n "$PHONE_DEMOS" ]]; then
  echo "== (c) phone-derived demos from $PHONE_DEMOS"
  if have "$DATA/phone_pan"; then echo "   exists, skipped"; else
    CAP=()
    if [[ "$MATCH_COUNTS" == "1" ]]; then CAP=(--max-episodes "$N_T"); fi
    # shellcheck disable=SC2086
    $PY train/convert_phone_npz.py $PHONE_DEMOS --like "$DATA/moka" --out "$DATA/phone_pan" --overwrite \
        ${CAP[@]+"${CAP[@]}"} ${PHONE_ARGS:-} 2>&1 | tee "$DATA/logs/phone_pan.log"
  fi
else
  echo "== (c) PHONE_DEMOS not set: phone demos and run P skipped for now"
fi

SOURCES=("$DATA/moka" "$DATA/teleop_pan")
[[ -f "$DATA/phone_pan/meta/info.json" ]] && SOURCES+=("$DATA/phone_pan")
echo "== check: sources mergeable and alike"
$PY train/check_datasets.py "${SOURCES[@]}" --png "$DATA/check_sources.png" --json "$DATA/check_sources.json" \
    2>&1 | tee "$DATA/logs/check_sources.log"

merge() {  # merge NAME PAN_DATASET
  local name=$1 pan=$2
  if have "$DATA/$name"; then echo "   $DATA/$name exists, skipped"; return; fi
  safe_rm "$DATA/$name"
  # One mp4 per source file (no stream concatenation): robust to encoder differences between sources.
  lerobot-edit-dataset --new_repo_id "local/cooking_twin_$name" --new_root "$DATA/$name" \
      --operation.type merge \
      --operation.repo_ids "['local/moka','local/$(basename "$pan")']" \
      --operation.roots "['$DATA/moka','$pan']" \
      --operation.concatenate_videos false 2>&1 | tee "$DATA/logs/merge_$name.log"
  if [[ "$PIN_STATS" == "1" ]]; then
    $PY train/pin_norm_stats.py --dataset "$DATA/$name" --policy "$POLICY" 2>&1 | tee "$DATA/logs/pin_$name.log"
  fi
}

echo "== (d) merge"
merge T "$DATA/teleop_pan"
if [[ -f "$DATA/phone_pan/meta/info.json" ]]; then merge P "$DATA/phone_pan"; fi

RUNS=()
for r in P T; do [[ -f "$DATA/$r/meta/info.json" ]] && RUNS+=("$DATA/$r"); done
echo "== check: merged datasets"
$PY train/check_datasets.py "${RUNS[@]}" --png "$DATA/check_runs.png" --json "$DATA/check_runs.json" \
    2>&1 | tee "$DATA/logs/check_runs.log"
echo "done: ${RUNS[*]}"
