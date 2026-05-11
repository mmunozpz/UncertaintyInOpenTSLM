#!/usr/bin/env bash
# run_tsqa_har_sleep_gpu_missing.sh
#
# Runs the missing-data MCSPU sweep for TSQA (stage 1), HAR (stage 3), and
# Sleep (stage 4) on GPU.  All four missing fractions are evaluated in a single
# forward-pass run per model/dataset and written to one JSONL file each.
#
# Usage:
#   bash run_tsqa_har_sleep_gpu_missing.sh
#   N_NOISE=50 MAX_SAMPLES=200 bash run_tsqa_har_sleep_gpu_missing.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

N_NOISE="${N_NOISE:-50}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
CLASS_BATCH="${CLASS_BATCH:-16}"
LLM_ID="${LLM_ID:-meta-llama/Llama-3.2-1B}"
MISSING_FRACTIONS=(0.1 0.25 0.5 0.75 1.0)
PYTHON="./venv/bin/python"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "TSQA + HAR + Sleep GPU missing-data sweep (N_NOISE=$N_NOISE  MAX_SAMPLES=$MAX_SAMPLES  CLASS_BATCH=$CLASS_BATCH)"
echo "Missing fractions: ${MISSING_FRACTIONS[*]}"
echo

# ---------------------------------------------------------------------------
# Helper: run one dataset (all missing fractions in a single call)
# ---------------------------------------------------------------------------
run_missing() {
    local dataset="$1"
    local model_type="$2"
    local ckpt="$3"
    local out_dir="$4"
    local out_file="$out_dir/${dataset}_${model_type}_missing.jsonl"
    local log_file="$out_dir/${dataset}_${model_type}_missing.log"

    if [[ -f "$out_file" ]]; then
        echo "  [skip] $dataset missing — output already exists ($out_file)"
        return 0
    fi

    mkdir -p "$out_dir"
    echo "  [start] $dataset missing fractions=${MISSING_FRACTIONS[*]} → $out_file"
    if "$PYTHON" compute_mcspu.py \
            --checkpoint        "$ckpt" \
            --model_type        "$model_type" \
            --llm_id            "$LLM_ID" \
            --dataset           "$dataset" \
            --perturbation_type missing \
            --missing_fractions "${MISSING_FRACTIONS[@]}" \
            --n_samples         "$N_NOISE" \
            --max_samples       "$MAX_SAMPLES" \
            --class_batch_size  "$CLASS_BATCH" \
            --device            cuda \
            --seed              42 \
            --output            "$out_file" \
            2>&1 | tee "$log_file"; then
        echo "  [done] $dataset missing"
    else
        echo "  [FAILED] $dataset missing — see $log_file"
        return 1
    fi
    echo
}

# ---------------------------------------------------------------------------
# Stage 1 — TSQA (4 fixed classes: A B C D)
# ---------------------------------------------------------------------------
echo "════════ Stage 1: TSQA ════════"
tsqa_failed=0
run_missing tsqa sp models/stage1_tsqa_sp.pt mcspu_results_missing || tsqa_failed=$?

# ---------------------------------------------------------------------------
# Stage 3 — HAR (8 fixed classes)
# ---------------------------------------------------------------------------
echo "════════ Stage 3: HAR ════════"
har_failed=0
run_missing har sp models/stage3_har_sp.pt mcspu_results_missing || har_failed=$?

# ---------------------------------------------------------------------------
# Stage 4 — Sleep (6 fixed classes)
# ---------------------------------------------------------------------------
echo "════════ Stage 4: Sleep ════════"
sleep_failed=0
run_missing sleep sp models/stage4_sleep_sp.pt mcspu_results_missing || sleep_failed=$?

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total_failed=$(( tsqa_failed + har_failed + sleep_failed ))
echo "══════════════════════════════════════════════"
echo "Done.  Failed: $total_failed total  (TSQA: $tsqa_failed  HAR: $har_failed  Sleep: $sleep_failed)"
echo "══════════════════════════════════════════════"
