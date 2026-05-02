#!/usr/bin/env bash
# run_ecg_cpu_parallel.sh
#
# Runs the 4 ECG sigma jobs in parallel on CPU, one process per sigma.
# Each process gets OMP_NUM_THREADS=<cores/4> to share the machine evenly.
#
# Usage:
#   bash run_ecg_cpu_parallel.sh
#   N_NOISE=50 MAX_SAMPLES=100 bash run_ecg_cpu_parallel.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

N_NOISE="${N_NOISE:-20}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
LLM_ID="${LLM_ID:-meta-llama/Llama-3.2-1B}"
SIGMAS=(0.1 0.5 1.0 2.0)
OUT_DIR="mcspu_results/stage5_ecg_sp"
CKPT="models/stage5_ecg_sp.pt"
PYTHON="./.venv_linux/bin/python"

# Give each of the 4 parallel processes an equal share of cores
TOTAL_CORES=$(nproc)
THREADS_PER_JOB=$(( TOTAL_CORES / ${#SIGMAS[@]} ))
THREADS_PER_JOB=$(( THREADS_PER_JOB < 1 ? 1 : THREADS_PER_JOB ))

mkdir -p "$OUT_DIR"

echo "ECG CPU parallel sweep: ${#SIGMAS[@]} sigmas in parallel"
echo "  Cores: $TOTAL_CORES  →  $THREADS_PER_JOB threads/job"
echo "  N_NOISE=$N_NOISE  MAX_SAMPLES=$MAX_SAMPLES"
echo

pids=()
for sigma in "${SIGMAS[@]}"; do
    out_file="$OUT_DIR/sigma_${sigma}.jsonl"

    if [[ -f "$out_file" ]]; then
        echo "  [skip] sigma=$sigma — output already exists"
        continue
    fi

    echo "  [start] sigma=$sigma → $out_file"
    OMP_NUM_THREADS=$THREADS_PER_JOB \
    MKL_NUM_THREADS=$THREADS_PER_JOB \
    "$PYTHON" compute_mcspu.py \
        --checkpoint "$CKPT" \
        --model_type sp \
        --llm_id    "$LLM_ID" \
        --dataset   ecg_qa \
        --n_samples "$N_NOISE" \
        --max_samples "$MAX_SAMPLES" \
        --sigma     "$sigma" \
        --device    cpu \
        --seed      42 \
        --output    "$out_file" \
        > "$OUT_DIR/sigma_${sigma}.log" 2>&1 &
    pids+=($!)
done

if [[ ${#pids[@]} -eq 0 ]]; then
    echo "Nothing to run — all outputs already exist."
    exit 0
fi

echo
echo "Waiting for ${#pids[@]} job(s) to finish..."
failed=0
for i in "${!pids[@]}"; do
    sigma="${SIGMAS[$i]}"
    if wait "${pids[$i]}"; then
        echo "  [done] sigma=$sigma"
    else
        echo "  [FAILED] sigma=$sigma — see $OUT_DIR/sigma_${sigma}.log"
        failed=$(( failed + 1 ))
    fi
done

echo
echo "══════════════════════════════════════════════"
echo "Done.  Failed: $failed / ${#pids[@]}"
echo "Results in: $OUT_DIR/"
echo "══════════════════════════════════════════════"
