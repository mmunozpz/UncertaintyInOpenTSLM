#!/usr/bin/env bash
# run_ecg_gpu_batched.sh
#
# Runs the 4 ECG sigma experiments sequentially on GPU using chunked candidate
# batching to avoid OOM.  The root cause of the previous OOM was scoring all
# answer candidates (up to 60 for ECG QA) in one forward pass over a very long
# ECG prompt (~3000 tokens), which demanded >32 GB of attention memory.
# --class_batch_size 4 limits each pass to 4 candidates at a time (~2 GB).
#
# Usage:
#   bash run_ecg_gpu_batched.sh
#   N_NOISE=50 MAX_SAMPLES=200 CLASS_BATCH=8 bash run_ecg_gpu_batched.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

N_NOISE="${N_NOISE:-50}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
CLASS_BATCH="${CLASS_BATCH:-8}"
LLM_ID="${LLM_ID:-meta-llama/Llama-3.2-1B}"
SIGMAS=(0.1 0.5 1.0 2.0)
OUT_DIR="mcspu_results/stage5_ecg_sp"
CKPT="models/stage5_ecg_sp.pt"
PYTHON="./.venv_linux/bin/python"

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT_DIR"

echo "ECG GPU batched sweep: ${#SIGMAS[@]} sigmas (sequential)"
echo "  CLASS_BATCH=$CLASS_BATCH  N_NOISE=$N_NOISE  MAX_SAMPLES=$MAX_SAMPLES"
echo "  CKPT=$CKPT"
echo

failed=0
for sigma in "${SIGMAS[@]}"; do
    out_file="$OUT_DIR/sigma_${sigma}.jsonl"
    log_file="$OUT_DIR/sigma_${sigma}.log"

    if [[ -f "$out_file" ]]; then
        echo "  [skip] sigma=$sigma — output already exists: $out_file"
        continue
    fi

    echo "  [start] sigma=$sigma → $out_file"
    if "$PYTHON" compute_mcspu.py \
            --checkpoint        "$CKPT" \
            --model_type        sp \
            --llm_id            "$LLM_ID" \
            --dataset           ecg_qa \
            --n_samples         "$N_NOISE" \
            --max_samples       "$MAX_SAMPLES" \
            --sigma             "$sigma" \
            --class_batch_size  "$CLASS_BATCH" \
            --device            cuda \
            --seed              42 \
            --output            "$out_file" \
            2>&1 | tee "$log_file"; then
        echo "  [done]   sigma=$sigma"
    else
        echo "  [FAILED] sigma=$sigma — see $log_file"
        failed=$(( failed + 1 ))
    fi
    echo
done

echo "══════════════════════════════════════════════"
echo "Done.  Failed: $failed / ${#SIGMAS[@]}"
echo "Results in: $OUT_DIR/"
echo "══════════════════════════════════════════════"
