#!/usr/bin/env bash
# run_mcspu_sweep.sh
#
# Full MCSPU sigma sweep across all supported stage checkpoints.
# Runs compute_mcspu.py for every (model, sigma) combination and writes
# results to mcspu_results/<tag>/sigma_<sigma>.jsonl
#
# Usage:
#   bash run_mcspu_sweep.sh                  # defaults below
#   N_NOISE=50 MAX_SAMPLES=100 bash run_mcspu_sweep.sh  # override env vars

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ── Tunable parameters ──────────────────────────────────────────────────────
# N_NOISE     : noise realizations per test sample (N in MCSPU)
#               20 is sufficient for HAR-vs-ECG discrimination; use 50 for publication.
# MAX_SAMPLES : how many test-set samples to score per run (leave empty = all)
#               On CPU, budget ~5 s × N_NOISE × MAX_SAMPLES per combo.
N_NOISE="${N_NOISE:-20}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
DEVICE="${DEVICE:-cpu}"
LLM_ID="${LLM_ID:-meta-llama/Llama-3.2-1B}"
SIGMAS=(0.1 0.5 1.0 2.0)
OUT_ROOT="${OUT_ROOT:-mcspu_results}"
# ────────────────────────────────────────────────────────────────────────────

# Model/dataset combos — format: "checkpoint_file|model_type|dataset|output_tag"
# Stage 2 (M4) is excluded: captioning task, no MCQ answer vocab.
COMBOS=(
  "models/stage1_tsqa_sp.pt|sp|tsqa|stage1_tsqa_sp"
  "models/stage3_har_sp.pt|sp|har|stage3_har_sp"
  "models/stage4_sleep_sp.pt|sp|sleep|stage4_sleep_sp"
  "models/stage5_ecg_sp.pt|sp|ecg_qa|stage5_ecg_sp"
)

mkdir -p "$OUT_ROOT"

total=$(( ${#COMBOS[@]} * ${#SIGMAS[@]} ))
run=0
skipped=0
failed=0

echo "MCSPU sweep: ${#COMBOS[@]} models × ${#SIGMAS[@]} sigmas = $total runs"
echo "  N_NOISE=$N_NOISE  MAX_SAMPLES=$MAX_SAMPLES  DEVICE=$DEVICE"
echo

for combo in "${COMBOS[@]}"; do
  IFS='|' read -r ckpt model_type dataset tag <<< "$combo"

  if [[ ! -f "$ckpt" ]]; then
    echo "⚠  Checkpoint not found, skipping all sigmas for $tag: $ckpt"
    skipped=$(( skipped + ${#SIGMAS[@]} ))
    run=$(( run + ${#SIGMAS[@]} ))
    continue
  fi

  out_dir="$OUT_ROOT/$tag"
  mkdir -p "$out_dir"

  for sigma in "${SIGMAS[@]}"; do
    run=$(( run + 1 ))
    out_file="$out_dir/sigma_${sigma}.jsonl"

    echo "──────────────────────────────────────────────────────────────"
    echo "[$run/$total]  $tag  sigma=$sigma"
    echo "  output: $out_file"
    echo "──────────────────────────────────────────────────────────────"

    if [[ -f "$out_file" ]]; then
      echo "  [skip] output already exists — delete to re-run"
      echo
      skipped=$(( skipped + 1 ))
      continue
    fi

    if ./$(dirname "${BASH_SOURCE[0]}")/.venv_linux/bin/python compute_mcspu.py \
        --checkpoint "$ckpt" \
        --model_type "$model_type" \
        --llm_id    "$LLM_ID" \
        --dataset   "$dataset" \
        --n_samples "$N_NOISE" \
        --max_samples "$MAX_SAMPLES" \
        --sigma     "$sigma" \
        --device    "$DEVICE" \
        --seed      42 \
        --output    "$out_file"; then
      echo
    else
      echo "  [FAILED] $tag sigma=$sigma — continuing with next run"
      failed=$(( failed + 1 ))
      echo
    fi
  done
done

echo "══════════════════════════════════════════════════════════════"
echo "Sweep complete."
echo "  Total runs  : $total"
echo "  Skipped     : $skipped"
echo "  Failed      : $failed"
echo "  Succeeded   : $(( total - skipped - failed ))"
echo "  Results in  : $OUT_ROOT/"
echo "══════════════════════════════════════════════════════════════"
