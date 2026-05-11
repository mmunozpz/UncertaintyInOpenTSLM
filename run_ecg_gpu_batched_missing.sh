#!/usr/bin/env bash
# run_ecg_gpu_batched_missing.sh
#
# Runs the missing-data MCSPU sweep for ECG (stage 5) on GPU using chunked
# candidate batching to avoid OOM (same CLASS_BATCH reasoning as the sigma
# version: up to 60 candidates × long ECG prompts needs small chunks).
# All missing fractions are swept in a single compute_mcspu.py call.
#
# Usage:
#   bash run_ecg_gpu_batched_missing.sh
#   N_NOISE=50 MAX_SAMPLES=200 CLASS_BATCH=4 bash run_ecg_gpu_batched_missing.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

N_NOISE="${N_NOISE:-50}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
CLASS_BATCH="${CLASS_BATCH:-4}"
LLM_ID="${LLM_ID:-meta-llama/Llama-3.2-1B}"
MISSING_FRACTIONS=(0.1 0.25 0.5 0.75 1.0)
OUT_DIR="mcspu_results_missing"
CKPT="models/stage5_ecg_sp.pt"
PYTHON="./venv/bin/python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_FILE="$OUT_DIR/ecg_sp_missing.jsonl"
LOG_FILE="$OUT_DIR/ecg_sp_missing.log"

mkdir -p "$OUT_DIR"

echo "ECG GPU batched missing-data sweep"
echo "  MISSING_FRACTIONS=${MISSING_FRACTIONS[*]}"
echo "  CLASS_BATCH=$CLASS_BATCH  N_NOISE=$N_NOISE  MAX_SAMPLES=$MAX_SAMPLES"
echo "  CKPT=$CKPT"
echo

if [[ -f "$OUT_FILE" ]]; then
    echo "  [skip] output already exists: $OUT_FILE"
    exit 0
fi

echo "  [start] missing fractions=${MISSING_FRACTIONS[*]} → $OUT_FILE"
if "$PYTHON" compute_mcspu.py \
        --checkpoint        "$CKPT" \
        --model_type        sp \
        --llm_id            "$LLM_ID" \
        --dataset           ecg_qa \
        --perturbation_type missing \
        --missing_fractions "${MISSING_FRACTIONS[@]}" \
        --n_samples         "$N_NOISE" \
        --max_samples       "$MAX_SAMPLES" \
        --class_batch_size  "$CLASS_BATCH" \
        --device            cuda \
        --seed              42 \
        --output            "$OUT_FILE" \
        2>&1 | tee "$LOG_FILE"; then
    echo "  [done] ECG missing sweep"
else
    echo "  [FAILED] ECG missing sweep — see $LOG_FILE"
    exit 1
fi

echo "══════════════════════════════════════════════"
echo "Done.  Results in: $OUT_FILE"
echo "══════════════════════════════════════════════"
