#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Evaluate OpenTSLMFlamingo on HAR CoT dataset with signal contribution tracking.

This script runs inference on the HAR test set and measures:
1. Per-label accuracy and Macro-F1 metrics
2. Signal contribution metrics (residual_stream, gated_cross_attn_output, signal_contribution_pct)

Usage:
    python evaluate_har_flamingo.py --checkpoint path/to/best_model.pt [--max_samples 100] [--use_noise]

Output:
    - Accuracy and F1 metrics per activity label
    - Signal contribution summary showing how much the accelerometer signal influences model output
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path for imports
script_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(script_dir / 'src'))
sys.path.insert(0, str(script_dir / 'src' / 'open_flamingo'))

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE

VALID_LABELS = [
    "biking", "lying", "running", "sitting",
    "standing", "walking", "walking_down", "walking_up",
]


def setup_device():
    """Setup the device for model inference."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device


def load_model(checkpoint_path: str, device: str, llm_id: str = "meta-llama/Llama-3.2-1B"):
    """Load the trained OpenTSLMFlamingo model."""
    print(f"Loading model from {checkpoint_path}...")

    model = OpenTSLMFlamingo(
        device=device,
        llm_id=llm_id,
        cross_attn_every_n_layers=1,
    )

    model.load_from_file(checkpoint_path)
    model.eval()
    print("Model loaded successfully")
    return model


def extract_answer(text: str) -> str:
    """Extract the activity label from model output.

    Looks for "Answer: <label>" pattern, falls back to last word.
    """
    if text is None:
        return ""
    pred = text.strip()
    # Find the last occurrence of 'Answer:' (case-insensitive)
    match = list(re.finditer(r"answer:\s*", pred, re.IGNORECASE))
    if match:
        start = match[-1].end()
        label = pred[start:].strip()
    else:
        label = pred.split()[-1] if pred.split() else ""
    # Remove trailing punctuation
    label = re.sub(r"[\.,;:!?]+$", "", label)
    return label.lower().strip()


def normalize_label(label: str) -> str:
    """Lowercase and strip."""
    if label is None:
        return ""
    return label.lower().strip()


def run_evaluation(
    model: OpenTSLMFlamingo,
    dataset: HARCoTQADataset,
    max_samples: int = None,
    max_new_tokens: int = 400,
) -> Dict[str, Any]:
    """Run evaluation on the dataset with signal tracking."""

    # Enable signal tracking
    model.enable_signal_tracking()
    model.clear_signal_measurements()

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        )
    )

    results = []
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

    print(f"\nRunning inference on {num_samples} samples...")
    print("=" * 70)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, total=num_samples, desc="Evaluating")):
            if idx >= num_samples:
                break

            try:
                sample = batch[0]

                # Generate prediction
                predictions = model.generate(batch, max_new_tokens=max_new_tokens)
                prediction = predictions[0] if predictions else ""

                # Get ground truth
                ground_truth = sample.get("answer", "")
                gt_label = sample.get("label", "")

                # Extract and compare
                pred_answer = extract_answer(prediction)
                gt_answer = normalize_label(gt_label)
                is_correct = int(pred_answer == gt_answer)

                result = {
                    "sample_idx": idx,
                    "ground_truth_label": gt_answer,
                    "ground_truth_full": ground_truth,
                    "prediction": prediction,
                    "pred_answer": pred_answer,
                    "accuracy": is_correct,
                }
                results.append(result)

                # Print first few samples
                if idx < 3:
                    print(f"\nSample {idx + 1}:")
                    print(f"  Ground truth: {gt_answer}")
                    print(f"  Prediction: {pred_answer}")
                    print(f"  Correct: {is_correct}")

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Disable tracking
    model.disable_signal_tracking()

    return {
        "results": results,
        "signal_contribution": model.get_signal_contribution_summary(),
    }


def calculate_aggregate_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Calculate aggregate metrics including per-class F1 and Macro-F1."""
    if not results:
        return {}

    total_correct = sum(r["accuracy"] for r in results)
    total_samples = len(results)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    # Per-class F1
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in results:
        gt = r["ground_truth_label"]
        pred = r["pred_answer"]
        if pred == gt:
            class_stats[gt]["tp"] += 1
        else:
            class_stats[gt]["fn"] += 1
            class_stats[pred]["fp"] += 1

    per_class = {}
    f1_sum = 0
    valid_classes = 0
    for class_name, stats in class_stats.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[class_name] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "support": tp + fn,
        }
        if tp + fn > 0:
            f1_sum += f1
            valid_classes += 1

    macro_f1 = f1_sum / valid_classes if valid_classes > 0 else 0

    return {
        "overall": {
            "total_samples": total_samples,
            "total_correct": total_correct,
            "accuracy": overall_accuracy,
            "macro_f1": macro_f1,
        },
        "per_class": per_class,
    }


def print_metrics_table(signal_summary: Dict, aggregate_metrics: Dict):
    """Print the compact metrics table: G, A, R, G*A/R, F1, Accuracy."""
    overall = aggregate_metrics.get("overall", {})
    accuracy = overall.get("accuracy", 0)
    macro_f1 = overall.get("macro_f1", 0)

    sig = signal_summary.get("overall", {})
    G = abs(sig.get("attn_gate_tanh_mean", 0))
    A = sig.get("raw_attn_output_mean", 0)
    R = sig.get("residual_stream_mean", 0)
    est_contrib = (G * A / R * 100) if R > 0 else 0

    print("\n" + "=" * 90)
    print("METRICS SUMMARY")
    print("=" * 90)
    print(f"{'G=|avg tanh(gate)|':<22} {'A=avg raw_attn':<18} {'R=avg residual':<18} {'G*A/R (%)':<14} {'F1':<10} {'Accuracy':<10}")
    print("-" * 90)
    print(f"{G:<22.6f} {A:<18.4f} {R:<18.4f} {est_contrib:<14.4f} {macro_f1:<10.4f} {accuracy:<10.4f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenTSLMFlamingo on HAR CoT with signal tracking")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate (None for all)")
    parser.add_argument("--max_new_tokens", type=int, default=400, help="Max tokens to generate")
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B", help="LLM ID")
    parser.add_argument("--use_noise", action="store_true", help="Replace accelerometer signals with noise")
    parser.add_argument("--noise_type", type=str, default="gaussian", choices=["gaussian", "shuffle", "zero", "uniform"], help="Type of noise")
    parser.add_argument("--noise_seed", type=int, default=67, help="Seed for noise generation")
    parser.add_argument("--strip_stats", action="store_true", help="Strip mean/std from text descriptions when using noise")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    # Setup
    device = setup_device()

    # Configure noise mode
    if args.use_noise:
        print(f"[NOISE MODE] Signals will be replaced with {args.noise_type} noise (seed={args.noise_seed}, strip_stats={args.strip_stats})")
        HARCoTQADataset.set_noise_mode(use_noise=True, noise_type=args.noise_type, noise_seed=args.noise_seed, strip_stats=args.strip_stats)
    else:
        HARCoTQADataset.set_noise_mode(use_noise=False)

    # Load model
    model = load_model(args.checkpoint, device, args.llm_id)

    # Load dataset
    print("Loading HAR CoT dataset (test split)...")
    dataset = HARCoTQADataset(
        split="test",
        EOS_TOKEN=model.text_tokenizer.eos_token,
    )
    print(f"Loaded {len(dataset)} samples")

    # Run evaluation
    eval_results = run_evaluation(
        model,
        dataset,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )

    # Calculate aggregate metrics
    aggregate_metrics = calculate_aggregate_metrics(eval_results["results"])

    # Print compact metrics table
    print_metrics_table(eval_results["signal_contribution"], aggregate_metrics)

    # Print detailed per-class breakdown
    print(f"\nPer-Class Metrics:")
    for class_name, stats in sorted(aggregate_metrics.get("per_class", {}).items()):
        print(f"  {class_name}: F1={stats['f1']:.4f}, P={stats['precision']:.4f}, R={stats['recall']:.4f} (support={stats['support']})")

    # Print full signal contribution breakdown
    model.print_signal_contribution_summary()

    # Build compact metrics dict for JSON
    overall = aggregate_metrics.get("overall", {})
    sig = eval_results["signal_contribution"].get("overall", {})
    G = abs(sig.get("attn_gate_tanh_mean", 0))
    A = sig.get("raw_attn_output_mean", 0)
    R = sig.get("residual_stream_mean", 0)
    compact_metrics = {
        "G_abs_avg_tanh_gate": G,
        "A_avg_raw_attn_output": A,
        "R_avg_residual_stream": R,
        "est_signal_contribution_pct": (G * A / R * 100) if R > 0 else 0,
        "macro_f1": overall.get("macro_f1", 0),
        "accuracy": overall.get("accuracy", 0),
    }

    # Save results if output path specified
    if args.output:
        output_data = {
            "checkpoint": args.checkpoint,
            "use_noise": args.use_noise,
            "noise_type": args.noise_type if args.use_noise else None,
            "compact_metrics": compact_metrics,
            "aggregate_metrics": aggregate_metrics,
            "signal_contribution": eval_results["signal_contribution"],
        }
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return eval_results


if __name__ == "__main__":
    main()
