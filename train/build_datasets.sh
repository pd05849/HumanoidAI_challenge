#!/usr/bin/env bash
# Build the training datasets for run P (phone demos + moka) and run T (teleop demos + moka).
#
#   (a) moka subset   : LIBERO-10 task 2 episodes out of HuggingFaceVLA/libero      -> $DATA/moka
#   (b) teleop pan    : the 50 LIBERO-90 task 21 human demos, re-rendered at 256 px  -> $DATA/teleop_pan
#   (c) phone pan     : sim/make_demos.py episodes (from the phone-tracked paths)    -> $DATA/phone_pan
#   (c2) phone pan all: optional, every phone-derived demo (PHONE_ALL_DATASET)      -> $DATA/phone_pan_all
#   (d) merge         : P = moka + phone_pan, T = moka + teleop_pan                  -> $DATA/P, $DATA/T
#                       Pall = moka + phone_pan_all (if PHONE_ALL_DATASET is set)    -> $DATA/Pall
#       then the normalization stats of P and T are pinned to smolvla_libero's own (PIN_STATS=1).
#
# Usage (repo root, LeRobot env with lerobot[libero,smolvla], GPU machine):
#   PHONE_DEMOS=/path/to/make_demos_out bash train/build_datasets.sh      # npz episodes (sim/make_demos.py)
#   PHONE_DATASET=/path/to/phone_pan bash train/build_datasets.sh         # or a LeRobot dataset (folder or
#                                       Hugging Face dataset id, e.g. <hf_user>/libero90_frying_pan_phone_50) built
#                                       elsewhere with train/convert_phone_npz.py; if its fps label or
#                                       codec differs from moka's, it is re-written (conform_dataset.py)
# Without either, only moka, teleop_pan and T are built (run it again later for P).
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
PHONE_DATASET=${PHONE_DATASET:-}
PHONE_ALL_DATASET=${PHONE_ALL_DATASET:-}
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

fetch_dataset() {  # fetch_dataset SRC: echo a local folder for SRC (a folder, or a Hub dataset id to download)
  if [[ -d "$1" ]]; then echo "$1"; return; fi
  local dst="$DATA/hub/${1//\//__}"
  echo "   downloading the Hugging Face dataset $1 -> $dst" >&2
  $PY -c "import sys; from huggingface_hub import snapshot_download as d; d(sys.argv[1], repo_type='dataset', local_dir=sys.argv[2])" \
      "$1" "$dst" >&2 || { echo "   cannot download $1 (not a folder here, nor a Hugging Face dataset you can access)" >&2; return 1; }
  echo "$dst"
}

import_phone() {  # import_phone SRC NAME: copy a ready-made phone dataset and make it mergeable with moka
  local src name=$2
  src=$(fetch_dataset "$1")
  safe_rm "$DATA/$name"
  cp -r "$src" "$DATA/$name"
  # Metadata only (names, info keys, robot_type). It refuses if fps, codec, shapes or streams
  # differ; then the episodes are decoded and written again in moka's format (a few minutes).
  if ! $PY train/check_datasets.py "$DATA/moka" "$DATA/$name" --harmonize --no-stats \
      2>&1 | tee "$DATA/logs/$name.log"; then
    echo "   metadata alone cannot match moka: re-writing $name in moka's format (conform_dataset.py)"
    safe_rm "$DATA/$name"
    $PY train/conform_dataset.py "$src" --like "$DATA/moka" --out "$DATA/$name" --overwrite \
        2>&1 | tee -a "$DATA/logs/$name.log"
    $PY train/check_datasets.py "$DATA/moka" "$DATA/$name" --no-stats 2>&1 | tee -a "$DATA/logs/$name.log"
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
elif [[ -n "$PHONE_DATASET" ]]; then
  echo "== (c) phone-derived dataset from $PHONE_DATASET"
  if have "$DATA/phone_pan"; then echo "   exists, skipped"; else import_phone "$PHONE_DATASET" phone_pan; fi
else
  echo "== (c) neither PHONE_DEMOS nor PHONE_DATASET set: phone demos and run P skipped for now"
fi

if [[ -n "$PHONE_ALL_DATASET" ]]; then
  echo "== (c2) all phone-derived demos from $PHONE_ALL_DATASET"
  if have "$DATA/phone_pan_all"; then echo "   exists, skipped"; else import_phone "$PHONE_ALL_DATASET" phone_pan_all; fi
fi

SOURCES=("$DATA/moka" "$DATA/teleop_pan")
[[ -f "$DATA/phone_pan/meta/info.json" ]] && SOURCES+=("$DATA/phone_pan")
[[ -f "$DATA/phone_pan_all/meta/info.json" ]] && SOURCES+=("$DATA/phone_pan_all")
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
if [[ -f "$DATA/phone_pan_all/meta/info.json" ]]; then merge Pall "$DATA/phone_pan_all"; fi

RUNS=()
for r in P Pall T; do [[ -f "$DATA/$r/meta/info.json" ]] && RUNS+=("$DATA/$r"); done
echo "== check: merged datasets"
$PY train/check_datasets.py "${RUNS[@]}" --png "$DATA/check_runs.png" --json "$DATA/check_runs.json" \
    2>&1 | tee "$DATA/logs/check_runs.log"
echo "done: ${RUNS[*]}"
