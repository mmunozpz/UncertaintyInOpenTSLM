#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Compute Monte Carlo Signal Perturbation Uncertainty (MCSPU) scores.

Runs N+1 forward passes per test sample (1 clean + N with additive Gaussian
noise) and reports U_signal = mean KL( p_clean || p_noisy ).

Usage:
    python compute_mcspu.py \\
        --checkpoint results/Llama3_2_1B/OpenTSLMFlamingo/stage3_cot/checkpoints/best_model.pt \\
        --model_type flamingo \\
        --dataset har \\
        --n_samples 50 \\
        --sigma 1.0 \\
        --device cuda \\
        --output mcspu_har.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "open_flamingo"))

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Answer vocabulary defaults
# ---------------------------------------------------------------------------

HAR_LABELS = [
    "biking", "lying", "running", "sitting", "standing",
    "walking", "walking_down", "walking_up",
]
TSQA_LABELS = ["A", "B", "C", "D"]


def _resolve_answer_vocab(dataset_name: str, answer_vocab_arg: str | None) -> list[str] | None:
    """Return the default answer vocab for *dataset_name*, or None for ECG QA (per-sample)."""
    if answer_vocab_arg:
        return [v.strip() for v in answer_vocab_arg.split(",")]

    if dataset_name == "har":
        return HAR_LABELS
    if dataset_name == "tsqa":
        return TSQA_LABELS
    if dataset_name == "sleep":
        from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
        return SleepEDFCoTQADataset.get_labels()
    if dataset_name == "ecg_qa":
        # Per-sample vocab stored in sample["possible_answers"] by ECGQACoTQADataset.
        return None
    raise ValueError(f"Unknown dataset: {dataset_name!r}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(args) -> object:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_type.upper()} model from {args.checkpoint} on {device} ...")

    if args.model_type == "sp":
        from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
        model = OpenTSLMSP(llm_id=args.llm_id, device=device)
        model.load_from_file(args.checkpoint)
    elif args.model_type == "flamingo":
        from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
        model = OpenTSLMFlamingo(llm_id=args.llm_id, device=device)
        model.load_from_file(args.checkpoint)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type!r}. Choose 'sp' or 'flamingo'.")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(args, model):
    eos = model.get_eos_token()
    split = args.split

    if args.dataset == "har":
        from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
        return HARCoTQADataset(split=split, EOS_TOKEN=eos)
    if args.dataset == "sleep":
        from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
        return SleepEDFCoTQADataset(split=split, EOS_TOKEN=eos)
    if args.dataset == "ecg_qa":
        from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
        return ECGQACoTQADataset(split=split, EOS_TOKEN=eos)
    if args.dataset == "tsqa":
        from opentslm.time_series_datasets.TSQADataset import TSQADataset
        return TSQADataset(split=split, EOS_TOKEN=eos)
    raise ValueError(f"Unknown dataset: {args.dataset!r}")


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict], args, level_label: str = "") -> None:
    scores = [r["mcspu_score"] for r in results]
    correct = sum(1 for r in results if r.get("clean_pred") == r.get("ground_truth"))
    n = len(results)
    accuracy = correct / n if n else float("nan")

    confidences = [max(r["clean_probs"]) for r in results]
    uncertainties = [1.0 - c for c in confidences]
    try:
        corr = float(np.corrcoef(scores, uncertainties)[0, 1])
    except Exception:
        corr = float("nan")

    print("\n" + "=" * 60)
    print("MCSPU SUMMARY" + (f"  [{level_label}]" if level_label else ""))
    print("=" * 60)
    print(f"  Samples scored:       {n}")
    print(f"  Dataset:              {args.dataset} ({args.split})")
    print(f"  Model:                {args.model_type} / {args.llm_id}")
    print(f"  Perturbation type:    {args.perturbation_type}")
    if args.perturbation_type == "gaussian":
        print(f"  Noise sigma:          {args.sigma}")
    else:
        print(f"  Missing fraction:     {level_label or args.missing_fractions}")
    print(f"  N realizations:       {args.n_samples}")
    print(f"  Mean MCSPU score:     {np.mean(scores):.4f}")
    print(f"  Std  MCSPU score:     {np.std(scores):.4f}")
    print(f"  Clean accuracy:       {accuracy:.4f}  ({correct}/{n})")
    print(f"  Corr(MCSPU, 1-conf):  {corr:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute MCSPU scores for an OpenTSLM checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--model_type", required=True, choices=["sp", "flamingo"],
        help="Model architecture"
    )
    parser.add_argument(
        "--llm_id", default="meta-llama/Llama-3.2-1B",
        help="HuggingFace LLM model ID (default: meta-llama/Llama-3.2-1B)"
    )
    parser.add_argument(
        "--dataset", required=True, choices=["har", "sleep", "ecg_qa", "tsqa"],
        help="Dataset to evaluate"
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "validation", "test"],
        help="Dataset split (default: test)"
    )
    parser.add_argument(
        "--answer_vocab",
        help="Comma-separated answer classes (overrides dataset default)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=50,
        help="Number of perturbation draws N per sample (default: 50)"
    )
    # gaussian args
    parser.add_argument(
        "--perturbation_type", default="gaussian", choices=["gaussian", "missing"],
        help="Perturbation type: gaussian (additive noise) or missing (random timepoint masking)"
    )
    parser.add_argument(
        "--sigma", type=float, default=1.0,
        help="[gaussian] Additive noise std deviation (default: 1.0)"
    )
    # missing data args
    parser.add_argument(
        "--missing_fractions", type=float, nargs="+",
        default=[0.1, 0.25, 0.5, 0.75, 1.0],
        help="[missing] Fraction(s) of timepoints to zero per channel. "
             "Multiple values sweep all levels in one run. (default: 0.1 0.25 0.5 0.75 1.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap on the number of test samples (default: all)"
    )
    parser.add_argument(
        "--output", default="mcspu_results.jsonl",
        help="Output JSONL file (default: mcspu_results.jsonl)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: cuda | cpu | mps (default: auto-detect)"
    )
    parser.add_argument(
        "--class_batch_size", type=int, default=4,
        help="Max answer candidates per GPU forward pass (default: 4). "
             "Reduce if OOM; increase to speed up (safe up to ~8 for ECG)."
    )
    args = parser.parse_args()

    model = _load_model(args)
    dataset = _load_dataset(args, model)
    answer_vocab = _resolve_answer_vocab(args.dataset, args.answer_vocab)

    if answer_vocab is None and args.dataset != "ecg_qa":
        parser.error(
            f"No default answer vocab for dataset {args.dataset!r}. "
            "Supply --answer_vocab."
        )

    from opentslm.uncertainty.mcspu import MCSpUScorer

    all_results: list[dict] = []

    if args.perturbation_type == "gaussian":
        levels = [args.sigma]
    else:
        levels = sorted(args.missing_fractions)

    for level in levels:
        if args.perturbation_type == "gaussian":
            print(f"\nScoring {args.dataset} {args.split} — gaussian σ={level}  N={args.n_samples} ...")
            scorer = MCSpUScorer(
                model=model,
                answer_vocab=answer_vocab or [],
                n_samples=args.n_samples,
                sigma=level,
                perturbation_type="gaussian",
                seed=args.seed,
                class_batch_size=args.class_batch_size,
            )
            level_label = f"σ={level}"
        else:
            print(f"\nScoring {args.dataset} {args.split} — missing fraction={level:.2f}  N={args.n_samples} ...")
            scorer = MCSpUScorer(
                model=model,
                answer_vocab=answer_vocab or [],
                n_samples=args.n_samples,
                missing_fraction=level,
                perturbation_type="missing",
                seed=args.seed,
                class_batch_size=args.class_batch_size,
            )
            level_label = f"missing={level:.2f}"

        results = scorer.score_dataset(dataset, max_samples=args.max_samples)

        for r in results:
            r["checkpoint"] = args.checkpoint
            r["dataset"] = args.dataset
            r["split"] = args.split
            r["llm_id"] = args.llm_id
            r["model_type"] = args.model_type

        all_results.extend(results)
        _print_summary(results, args, level_label=level_label)

    with open(args.output, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nAll results written to {args.output}  ({len(all_results)} records)")


if __name__ == "__main__":
    main()
