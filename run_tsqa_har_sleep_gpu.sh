#!/usr/bin/env bash
# run_har_sleep_gpu.sh
#
# Runs the 4-sigma MCSPU sweep for TSQA (stage 1), HAR (stage 3), and
# Sleep (stage 4) on GPU.  All three have fixed small vocabularies
# (4 / 8 / 6 classes), so OOM is not a concern and class_batch_size=8
# processes most in a single forward pass.
# Sigmas run sequentially within each dataset; datasets run sequentially.
#
# Usage:
#   bash run_har_sleep_gpu.sh
#   N_NOISE=50 MAX_SAMPLES=200 bash run_har_sleep_gpu.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

N_NOISE="${N_NOISE:-50}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
CLASS_BATCH="${CLASS_BATCH:-8}"
LLM_ID="${LLM_ID:-meta-llama/Llama-3.2-1B}"
SIGMAS=(0.1 0.5 1.0 2.0)
PYTHON="./.venv_linux/bin/python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "TSQA + HAR + Sleep GPU sweep (N_NOISE=$N_NOISE  MAX_SAMPLES=$MAX_SAMPLES  CLASS_BATCH=$CLASS_BATCH)"
echo

# ---------------------------------------------------------------------------
# Helper: run one dataset / sigma combination
# ---------------------------------------------------------------------------
run_sweep() {
    local dataset="$1"
    local model_type="$2"
    local ckpt="$3"
    local out_dir="$4"

    mkdir -p "$out_dir"
    local failed=0

    for sigma in "${SIGMAS[@]}"; do
        local out_file="$out_dir/sigma_${sigma}.jsonl"
        local log_file="$out_dir/sigma_${sigma}.log"

        if [[ -f "$out_file" ]]; then
            echo "  [skip] $dataset sigma=$sigma — output already exists"
            continue
        fi

        echo "  [start] $dataset sigma=$sigma → $out_file"
        if "$PYTHON" compute_mcspu.py \
                --checkpoint        "$ckpt" \
                --model_type        "$model_type" \
                --llm_id            "$LLM_ID" \
                --dataset           "$dataset" \
                --n_samples         "$N_NOISE" \
                --max_samples       "$MAX_SAMPLES" \
                --sigma             "$sigma" \
                --class_batch_size  "$CLASS_BATCH" \
                --device            cuda \
                --seed              42 \
                --output            "$out_file" \
                2>&1 | tee "$log_file"; then
            echo "  [done]   $dataset sigma=$sigma"
        else
            echo "  [FAILED] $dataset sigma=$sigma — see $log_file"
            failed=$(( failed + 1 ))
        fi
        echo
    done

    return $failed
}

# ---------------------------------------------------------------------------
# Stage 1 — TSQA (4 fixed classes: A B C D)
# ---------------------------------------------------------------------------
echo "════════ Stage 1: TSQA ════════"
tsqa_failed=0
run_sweep tsqa sp models/stage1_tsqa_sp.pt mcspu_results/stage1_tsqa_sp || tsqa_failed=$?

# ---------------------------------------------------------------------------
# Stage 3 — HAR (8 fixed classes)
# ---------------------------------------------------------------------------
echo "════════ Stage 3: HAR ════════"
har_failed=0
run_sweep har sp models/stage3_har_sp.pt mcspu_results/stage3_har_sp || har_failed=$?

# ---------------------------------------------------------------------------
# Stage 4 — Sleep (6 fixed classes)
# ---------------------------------------------------------------------------
echo "════════ Stage 4: Sleep ════════"
sleep_failed=0
run_sweep sleep sp models/stage4_sleep_sp.pt mcspu_results/stage4_sleep_sp || sleep_failed=$?

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total_failed=$(( tsqa_failed + har_failed + sleep_failed ))
echo "══════════════════════════════════════════════"
echo "Done.  Failed: $total_failed total  (TSQA: $tsqa_failed  HAR: $har_failed  Sleep: $sleep_failed)"
echo "══════════════════════════════════════════════"
