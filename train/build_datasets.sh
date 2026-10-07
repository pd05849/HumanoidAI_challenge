#!/usr/bin/env bash
# Build the training datasets of the three fine-tunes T, P and Pall (see docs/training.md).
#
#   (a) moka subset   : the 41 LIBERO-10 task 2 episodes in HuggingFaceVLA/libero    -> $DATA/moka
#   (b) teleop pan    : the 50 LIBERO-90 task 21 human demos, re-rendered at 256 px  -> $DATA/teleop_pan
#   (c) phone pan     : 50 phone-derived demos (PHONE_DATASET, or PHONE_DEMOS below) -> $DATA/phone_pan
#   (c2) phone pan all: all 153 phone-derived demos (PHONE_ALL_DATASET)              -> $DATA/phone_pan_all
#   (d) merge         : T = moka + teleop_pan, P = moka + phone_pan                  -> $DATA/T, $DATA/P
#                       Pall = moka + phone_pan_all                                  -> $DATA/Pall
#       then the state/action normalization stats of every merged dataset are pinned to
#       smolvla_libero's own (PIN_STATS=1).
#
# Usage, as run for the results (repo root, LeRobot 0.6.1 env with lerobot[libero,smolvla,training];
# MUJOCO_GL=osmesa where NVIDIA's EGL vendor file is missing, egl otherwise; no GPU needed):
#   PHONE_DATASET=Delcastillo8/libero90_frying_pan_phone_50 \
#   PHONE_ALL_DATASET=Delcastillo8/libero90_frying_pan_phone bash train/build_datasets.sh
# Give both: they have no defaults. Without PHONE_DATASET there is no P, without PHONE_ALL_DATASET
# no Pall. Each is a LeRobot dataset folder (e.g. datasets/phone_pan_P and datasets/phone_pan_all
# from README step 4) or a Hugging Face dataset id. If its features, fps label or codec differ
# from moka's (the Hub datasets store video, moka stores images in parquet), it is re-written in
# moka's format (conform_dataset.py).
# Expected: moka 41 episodes / 10,866 frames; teleop 50 of 50 kept; T 91 episodes / 24,801 frames;
# P 91; Pall 194; "ALL COMPATIBLE" at the end of $DATA/logs/check_sources.log.
#
# Alternative for P: PHONE_DEMOS="demos/r0 demos/r1 demos/r2" (sim/make_demos.py --out folders with
# episodes/*.npz) converts the npz episodes here with convert_phone_npz.py, capped at the number of
# teleop episodes and taken round-robin over the takes (MATCH_COUNTS=1). It wins over PHONE_DATASET
# if both are set. It does not make Pall: that still needs PHONE_ALL_DATASET.
# Without PHONE_DEMOS and PHONE_DATASET there is no phone_pan and no P; Pall needs only PHONE_ALL_DATASET.
# Steps whose output folder already exists are skipped; REBUILD=1 rebuilds everything.
#
# Environment knobs:
#   DATA=outputs/datasets  SOURCE_REPO=HuggingFaceVLA/libero  POLICY=HuggingFaceVLA/smolvla_libero
#   PIN_STATS=1  MATCH_COUNTS=1 (PHONE_DEMOS only: cap the phone episodes at the teleop count)
#   TELEOP_JOBS=4 (parallel processes replaying the 50 teleop demos; ~2 GB RAM each; one log per
#     part in $DATA/logs/teleop_pan.part*.log, one merged $DATA/teleop_pan_report.json;
#     TELEOP_JOBS=1 runs one process that logs to $DATA/logs/teleop_pan.log)
#   TELEOP_HDF5=/path/to/the/downloaded.hdf5 (skip its download)  TELEOP_DEMOS=0-9 (a subset, for tests)
#   TELEOP_ARGS="..." PHONE_ARGS="..." (extra flags for the converters; PHONE_ARGS: PHONE_DEMOS only)
#   PY=python  MUJOCO_GL=egl (default; osmesa renders on the CPU)
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
TELEOP_JOBS=${TELEOP_JOBS:-4}
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
if have "$DATA/teleop_pan"; then echo "   exists, skipped"
elif [[ "$TELEOP_JOBS" -le 1 || "${TELEOP_ARGS:-}" == *--demos* || "${TELEOP_ARGS:-}" == *--hdf5* ]]; then
  # shellcheck disable=SC2086
  $PY train/convert_teleop_hdf5.py --like "$DATA/moka" --out "$DATA/teleop_pan" --overwrite --video \
      ${TELEOP_ARGS:-} 2>&1 | tee "$DATA/logs/teleop_pan.log"
else
  # Each demo is replayed in the simulator (~2 min each on one core): split the demos over
  # TELEOP_JOBS processes, then merge the parts in demo order.
  H=${TELEOP_HDF5:-$($PY -c "import sys; sys.path.insert(0, 'train'); import convert_teleop_hdf5 as c; from huggingface_hub import hf_hub_download as d; print(d(c.HF_REPO, c.HF_FILE, repo_type='dataset'))" | tail -1)}
  N_DEMOS=$($PY -c "import sys, h5py; print(len(h5py.File(sys.argv[1], 'r')['data']))" "$H")
  LO=0; HI=$(( N_DEMOS - 1 ))
  if [[ -n "${TELEOP_DEMOS:-}" ]]; then LO=${TELEOP_DEMOS%-*}; HI=${TELEOP_DEMOS#*-}; fi   # e.g. 0-9 (testing)
  NJ=$(( HI - LO + 1 ))
  PARTS="$DATA/.teleop_parts"; rm -rf "$PARTS"; mkdir -p "$PARTS"
  echo "   demos $LO-$HI in $TELEOP_JOBS parallel parts (logs: $DATA/logs/teleop_pan.part*.log)"
  pids=(); roots=(); ids=()
  for ((k = 0; k < TELEOP_JOBS; k++)); do
    a=$(( LO + k * NJ / TELEOP_JOBS )); b=$(( LO + (k + 1) * NJ / TELEOP_JOBS - 1 ))
    (( b >= a )) || continue
    vid=(); (( k == 0 )) && vid=(--video)
    # shellcheck disable=SC2086
    $PY train/convert_teleop_hdf5.py --hdf5 "$H" --like "$DATA/moka" --out "$PARTS/part$k" --overwrite \
        --demos "$a-$b" ${vid[@]+"${vid[@]}"} ${TELEOP_ARGS:-} > "$DATA/logs/teleop_pan.part$k.log" 2>&1 &
    pids+=($!); roots+=("'$PARTS/part$k'"); ids+=("'local/teleop_part$k'")
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  if (( fail )); then echo "   a teleop part failed: see $DATA/logs/teleop_pan.part*.log"; exit 1; fi
  safe_rm "$DATA/teleop_pan"
  lerobot-edit-dataset --new_repo_id local/teleop_pan --new_root "$DATA/teleop_pan" \
      --operation.type merge --operation.repo_ids "[$(IFS=,; echo "${ids[*]}")]" \
      --operation.roots "[$(IFS=,; echo "${roots[*]}")]" --operation.concatenate_videos false \
      > "$DATA/logs/teleop_pan.log" 2>&1
  # one report for all parts (per-demo replay mode, distance to the eval layouts, ...)
  $PY - "$DATA" "$PARTS" <<'PYEOF'
import glob, json, sys
data, parts = sys.argv[1], sys.argv[2]
reps = [json.load(open(p)) for p in sorted(glob.glob(f"{parts}/part*_report.json"), key=lambda p: int(p.rsplit("part", 1)[1].split("_")[0]))]
out = dict(reps[0]); out["demos"] = [d for r in reps for d in r.get("demos", [])]
for k in ("n_demos", "n_kept", "n_actions_mode", "n_states_mode", "n_frames", "over_eval_horizon", "wall_s"):
    if all(isinstance(r.get(k), (int, float)) for r in reps):
        out[k] = sum(r[k] for r in reps)
json.dump(out, open(f"{data}/teleop_pan_report.json", "w"), indent=1)
print(f"   merged {len(reps)} parts: {out.get('n_kept')} of {out.get('n_demos')} demos kept")
PYEOF
  cp "$PARTS/part0_episode0.mp4" "$DATA/teleop_pan_episode0.mp4" 2>/dev/null || true
  rm -rf "$PARTS"
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
