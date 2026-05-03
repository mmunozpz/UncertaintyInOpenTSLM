#!/usr/bin/env python3
"""
opentslm_uncertainty_test.py — MCSPU production gate for a single checkpoint

Scores one OpenTSLM checkpoint across four noise levels and decides whether the model is production-ready: it must pass all sanity tests and demonstrate signal sensitivity.  
Any model that ignores the signal fails the gate.

IMPORTANT: run with n_noise = 50 and max_samples = 200 (default), it can be changed but these values are the statistically most significant designed for this test.

Usage
-----
    python opentslm_uncertainty_test.py \\
        --checkpoint  models/my_har_model.pt \\
        --dataset     har \\
        --model_type  sp \\
        --device      cuda

    python opentslm_uncertainty_test.py \\
        --checkpoint  models/my_ecg_model.pt \\
        --dataset     ecg_qa \\
        --model_type  sp \\
        --llm_id      meta-llama/Llama-3.2-1B \\
        --device      cuda \\
        --n_noise     50 \\
        --max_samples 200 \\
        --class_batch 8 \\
        --sigmas      0.1 0.5 1.0 2.0 \\
        --out_dir     plots/

        
--dataset choices: tsqa | har | sleep | ecg_qa

Exit code
---------
    0  PRODUCTION READY  (all sanity + sensitivity tests pass)
    1  NOT PRODUCTION READY  (one or more tests failed)
"""

import matplotlib.pyplot as plt
import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

import matplotlib
matplotlib.use("Agg")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))
sys.path.insert(0, str(_HERE / "src" / "open_flamingo"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


STAGE_REGISTRY = {
    "tsqa": {
        "label":        "TSQA",
        "answer_vocab": ["A", "B", "C", "D"],
    },
    "har": {
        "label":        "HAR",
        "answer_vocab": [
            "biking", "lying", "running", "sitting",
            "standing", "walking", "walking_down", "walking_up",
        ],
    },
    "sleep": {
        "label":        "Sleep EDF",
        "answer_vocab": [
            "Wake", "Non-REM stage 1", "Non-REM stage 2",
            "Non-REM stage 3", "REM sleep", "Movement",
        ],
    },
    "ecg_qa": {
        "label":        "ECG-QA",
        "answer_vocab": None,   # per-sample vocab from dataset
    },
}

# TEST THRESHOLDS: See UNCERTAINTY_TEST_GUIDE.txt for full derivation from experimental data.

THRESHOLDS = {
    "min_mcspu_high_sigma": 0.05,   # mean MCSPU at σ_max  (nats)
    "min_mcspu_range":      0.02,   # MCSPU(σ_max) − MCSPU(σ_min)
    "max_mannwhitney_p":    0.05,   # one-sided p-value
    "min_cohens_d":         0.30,   # Cohen's d
    "kl_consistency_atol":  1e-5,   # |score − mean(kl)| per record
    "prob_sum_atol":        1e-3,   # |Σ probs − 1|
}

# MCSPU COMPUTATION


def _add_noise(sample: Dict[str, Any], sigma: float, rng: np.random.Generator) -> Dict[str, Any]:
    """Deep-copy sample and add Gaussian noise to time_series only."""
    noisy = copy.deepcopy(sample)
    ts = noisy["time_series"]
    if isinstance(ts, torch.Tensor):
        noise = torch.tensor(rng.normal(0.0, sigma, ts.shape),
                             dtype=ts.dtype, device=ts.device)
        noisy["time_series"] = ts + noise
    else:  # List[List[float]]
        noisy["time_series"] = [
            (np.asarray(ch) + rng.normal(0.0, sigma, len(ch))).tolist()
            for ch in ts
        ]
    return noisy


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    p, q = p.float(), q.float()
    # clamp: fp32 arithmetic can produce tiny negatives (~1e-7) that aren't real
    return float(max(0.0, torch.sum(p * (torch.log(p + eps) - torch.log(q + eps))).item()))


def _get_logprobs(model, sample: Dict[str, Any], vocab: List[str],
                  class_batch: int) -> torch.Tensor:
    """
    Collate *sample* and call model.compute_class_logprobs with chunked batching.
    Returns a 1-D CPU float32 tensor of shape (len(vocab),).
    """
    from opentslm.time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate,
    )
    collated = extend_time_series_to_match_patch_size_and_aggregate(
        [copy.deepcopy(sample)])[0]
    with torch.no_grad():
        return model.compute_class_logprobs(collated, vocab, class_batch_size=class_batch)


def score_sample(
    model,
    sample: Dict[str, Any],
    vocab: List[str],
    n_noise: int,
    sigma: float,
    class_batch: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Run N+1 forward passes and return the MCSPU record for one sample."""
    # clean pass
    clean_lp = _get_logprobs(model, sample, vocab, class_batch)
    p0 = torch.softmax(clean_lp.float(), dim=0)

    per_kl: List[float] = []
    for _ in range(n_noise):
        noisy = _add_noise(sample, sigma, rng)
        noisy_lp = _get_logprobs(model, noisy, vocab, class_batch)
        pi = torch.softmax(noisy_lp.float(), dim=0)
        per_kl.append(_kl(p0, pi))
        if model.device != "cpu":
            torch.cuda.empty_cache()

    return {
        "mcspu_score":   float(np.mean(per_kl)),
        "clean_logprobs": clean_lp.tolist(),
        "clean_probs":    p0.tolist(),
        "clean_pred":     vocab[int(torch.argmax(p0).item())],
        "per_noise_kl":   per_kl,
        "sigma":          sigma,
        "n_samples":      n_noise,
        "answer_vocab":   vocab,
    }


def score_dataset(
    model,
    dataset,
    default_vocab: Optional[List[str]],
    n_noise: int,
    sigma: float,
    class_batch: int,
    max_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = min(max_samples, len(dataset))
    results = []

    for idx in range(n):
        sample = dataset[idx]
        vocab = sample.get("possible_answers") or default_vocab
        result = score_sample(model, sample, vocab,
                              n_noise, sigma, class_batch, rng)
        result["sample_idx"] = idx
        result["ground_truth"] = sample.get("answer", "")
        results.append(result)

        if (idx + 1) % 10 == 0 or (idx + 1) == n:
            print(f"    [{idx+1:>4}/{n}]  mcspu={result['mcspu_score']:.4f}"
                  f"  pred={result['clean_pred']!r}", flush=True)

    return results


# MODEL & DATASET LOADING

def load_model(checkpoint: Path, llm_id: str, device: str, model_type: str = "sp"):
    print(
        f"  Loading {model_type.upper()} checkpoint {checkpoint.name} on {device} …", flush=True)
    if model_type == "sp":
        from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
        model = OpenTSLMSP(llm_id=llm_id, device=device)
    elif model_type == "flamingo":
        from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
        model = OpenTSLMFlamingo(llm_id=llm_id, device=device)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    model.load_from_file(str(checkpoint))
    model.eval()
    return model


def load_dataset_split(dataset_name: str, eos: str):
    if dataset_name == "tsqa":
        from opentslm.time_series_datasets.TSQADataset import TSQADataset
        return TSQADataset(split="test", EOS_TOKEN=eos)
    if dataset_name == "har":
        from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
        return HARCoTQADataset(split="test", EOS_TOKEN=eos)
    if dataset_name == "sleep":
        from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
        return SleepEDFCoTQADataset(split="test", EOS_TOKEN=eos)
    if dataset_name == "ecg_qa":
        from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
        return ECGQACoTQADataset(split="test", EOS_TOKEN=eos)
    raise ValueError(f"Unknown dataset: {dataset_name!r}")


# TEST SUITE

@dataclass
class TestResult:
    name: str
    passed: bool
    metric: float
    threshold: float
    op: str
    unit: str = ""
    details: str = ""

    def symbol(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class StageReport:
    stage_key: str
    label: str
    sanity: List[TestResult] = field(default_factory=list)
    sensitivity: List[TestResult] = field(default_factory=list)

    def sanity_ok(self) -> bool: return all(t.passed for t in self.sanity)

    def sensitivity_ok(
        self) -> bool: return all(t.passed for t in self.sensitivity)
    def production_ready(
        self) -> bool: return self.sanity_ok() and self.sensitivity_ok()


def run_sanity(sigma_data: Dict[float, List[dict]]) -> List[TestResult]:
    all_r = [r for records in sigma_data.values() for r in records]
    out = []

    # 1. no NaN/Inf
    bad = sum(1 for r in all_r if not np.isfinite(r["mcspu_score"]))
    out.append(TestResult("no_nan_inf_scores", bad == 0,
                          float(bad), 0, "==", "bad records",
                          f"{bad}/{len(all_r)} non-finite scores"))

    # 2. KL ≥ 0
    min_kl = min(kl for r in all_r for kl in r["per_noise_kl"])
    out.append(TestResult("kl_nonneg", min_kl >= -1e-9,
                          float(min_kl), 0.0, ">=", "nats",
                          f"min per_noise_kl = {min_kl:.2e}"))

    # 3. score == mean(kl)
    worst = max(abs(r["mcspu_score"] - np.mean(r["per_noise_kl"]))
                for r in all_r)
    thr = THRESHOLDS["kl_consistency_atol"]
    out.append(TestResult("score_consistency", worst <= thr,
                          float(worst), thr, "<=", "nats",
                          f"max |score − mean(kl)| = {worst:.2e}"))

    # 4. probs sum to 1
    worst_sum = max(abs(sum(r["clean_probs"]) - 1.0) for r in all_r)
    thr2 = THRESHOLDS["prob_sum_atol"]
    out.append(TestResult("probs_normalized", worst_sum <= thr2,
                          float(worst_sum), thr2, "<=", "",
                          f"max |Σ probs − 1| = {worst_sum:.2e}"))
    return out


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


def run_sensitivity(sigma_data: Dict[float, List[dict]],
                    sigma_lo: float, sigma_hi: float) -> List[TestResult]:
    lo = np.array([r["mcspu_score"] for r in sigma_data.get(sigma_lo, [])])
    hi = np.array([r["mcspu_score"] for r in sigma_data.get(sigma_hi, [])])
    if len(lo) == 0 or len(hi) == 0:
        return []
    out = []

    # 1. magnitude at high sigma
    mean_hi = float(hi.mean())
    thr = THRESHOLDS["min_mcspu_high_sigma"]
    out.append(TestResult("sensitivity_magnitude", mean_hi > thr,
                          mean_hi, thr, ">", "nats",
                          f"mean MCSPU(σ={sigma_hi}) = {mean_hi:.4f}"))

    # 2. absolute growth
    delta = float(hi.mean() - lo.mean())
    thr = THRESHOLDS["min_mcspu_range"]
    out.append(TestResult("sensitivity_range", delta > thr,
                          delta, thr, ">", "nats",
                          f"MCSPU({sigma_hi}) − MCSPU({sigma_lo}) = {delta:.4f}"))

    # 3. Mann-Whitney U (one-sided)
    _, p = stats.mannwhitneyu(hi, lo, alternative="greater")
    thr = THRESHOLDS["max_mannwhitney_p"]
    out.append(TestResult("statistical_significance", float(p) < thr,
                          float(p), thr, "<", "p-value",
                          f"Mann-Whitney U p = {p:.2e}"))

    # 4. Cohen's d
    d = _cohens_d(hi, lo)
    thr = THRESHOLDS["min_cohens_d"]
    out.append(TestResult("effect_size", d > thr,
                          d, thr, ">", "σ",
                          f"Cohen's d = {d:.2f}"))

    return out


# TERMINAL REPORT

_G, _R, _Y, _RST, _B = "\033[92m", "\033[91m", "\033[93m", "\033[0m", "\033[1m"


def _c(txt, col): return f"{col}{txt}{_RST}"
def _pass(p): return _c("PASS", _G) if p else _c("FAIL", _R)


def print_report(reports: List[StageReport]) -> int:
    rep = reports[0]
    div = "═" * 60
    print(f"\n{_B}{div}{_RST}")
    print(f"{_B}  MCSPU PRODUCTION GATE — {rep.label}{_RST}")
    print(f"{_B}{div}{_RST}\n")

    def _section(title, tests):
        w = max(len(t.name) for t in tests) + 2
        print(f"{_B}{title}{_RST}")
        print("  " + "─" * (w + 30))
        for t in tests:
            print(f"  {t.name:{w}s}  {_pass(t.passed)}  {t.details}")
        print()

    _section("SANITY TESTS", rep.sanity)
    _section("SIGNAL SENSITIVITY TESTS", rep.sensitivity)

    # threshold + metric table
    print("─" * 60)
    print(f"  {'Test':30s}  {'Threshold':>10}  {'Measured':>10}  Result")
    print("─" * 60)
    for t in rep.sensitivity:
        thr_str = f"{t.op} {t.threshold:.4f}"
        print(f"  {t.name:<30}  {thr_str:>10}  {t.metric:>10.4f}  {_pass(t.passed)}")

    # final verdict
    ready = rep.production_ready()
    print(f"\n{_B}VERDICT{_RST}")
    print("─" * 60)
    if ready:
        print(_c(f"  ✓  {rep.label} — PRODUCTION READY", _G))
    else:
        print(_c(f"  ✗  {rep.label} — NOT PRODUCTION READY", _R))
        if not rep.sanity_ok():
            print(_c("    Sanity tests failed: computation is broken.", _R))
        if not rep.sensitivity_ok():
            print(_c("    Model does not use the signal. Do not deploy.", _R))
    print()
    return int(not ready)

# PLOTS


PALETTE = {
    "TSQA":      "#4C72B0",
    "HAR":       "#DD8452",
    "Sleep EDF": "#55A868",
    "ECG-QA":   "#C44E52",
}
STYLE = {
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
}


def _save(fig, stem: Path):
    path = str(stem) + ".png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_mcspu_vs_sigma(reports, sigma_data_all, sigmas, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    for rep in reports:
        sdata = sigma_data_all[rep.stage_key]
        colour = PALETTE[rep.label]
        xs, ys, cis = [], [], []
        for s in sigmas:
            sc = np.array([r["mcspu_score"] for r in sdata.get(s, [])])
            if len(sc) == 0:
                continue
            xs.append(s)
            ys.append(sc.mean())
            cis.append(1.96 * sc.std() / np.sqrt(len(sc)))
        xs, ys, cis = np.array(xs), np.array(ys), np.array(cis)
        verdict = "PASS" if rep.sensitivity_ok() else "FAIL"
        ax.plot(xs, ys, marker="o", color=colour, linewidth=2,
                label=f"{rep.label}  [{verdict}]")
        ax.fill_between(xs, ys - cis, ys + cis, alpha=0.12, color=colour)
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Mean MCSPU (± 95% CI)")
    ax.set_title("Signal sensitivity vs noise magnitude")
    ax.set_xticks(sigmas)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    _save(fig, out_dir / "mcspu_vs_sigma")


def plot_distributions(reports, sigma_data_all, sigmas, out_dir):
    fig, axes = plt.subplots(1, len(sigmas), figsize=(
        4 * len(sigmas), 4.5), sharey=False)
    for col, sigma in enumerate(sigmas):
        ax = axes[col]
        scores_list, positions, colours = [], [], []
        for i, rep in enumerate(reports):
            sc = [r["mcspu_score"]
                  for r in sigma_data_all[rep.stage_key].get(sigma, [])]
            if not sc:
                continue
            scores_list.append(sc)
            positions.append(i + 1)
            colours.append(PALETTE[rep.label])
        vp = ax.violinplot(scores_list, positions=positions,
                           showmedians=True, showextrema=False)
        for body, col_ in zip(vp["bodies"], colours):
            body.set_facecolor(col_)
            body.set_alpha(0.7)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)
        ax.set_xticks(positions)
        ax.set_xticklabels([r.label for r in reports],
                           rotation=30, ha="right", fontsize=9)
        ax.set_title(f"σ = {sigma}")
        if col == 0:
            ax.set_ylabel("MCSPU score")
    fig.suptitle("Per-sample MCSPU distributions", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir / "distributions")


def plot_heatmap(reports, sigma_data_all, sigmas, out_dir):
    matrix = np.full((len(reports), len(sigmas)), np.nan)
    for i, rep in enumerate(reports):
        for j, s in enumerate(sigmas):
            sc = [r["mcspu_score"]
                  for r in sigma_data_all[rep.stage_key].get(s, [])]
            if sc:
                matrix[i, j] = np.mean(sc)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Mean MCSPU")
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f"σ={s}" for s in sigmas])
    ax.set_yticks(range(len(reports)))
    ax.set_yticklabels([r.label for r in reports])
    ax.set_title("Mean MCSPU (dataset × σ)")
    vmid = np.nanmean(matrix)
    for i in range(len(reports)):
        for j in range(len(sigmas)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                        fontweight="bold", color="white" if v > vmid else "black")
    fig.tight_layout()
    _save(fig, out_dir / "heatmap")


def plot_test_results(reports, out_dir):
    all_tests = ([(t, "sanity") for t in reports[0].sanity] +
                 [(t, "sensitivity") for t in reports[0].sensitivity])
    n_t, n_d = len(all_tests), len(reports)
    matrix = np.zeros((n_t, n_d))
    for j, rep in enumerate(reports):
        for i, (_, cat) in enumerate(all_tests):
            src = rep.sanity if cat == "sanity" else rep.sensitivity
            idx_ = i if cat == "sanity" else i - len(rep.sanity)
            matrix[i, j] = 1.0 if src[idx_].passed else 0.0

    fig, ax = plt.subplots(figsize=(max(6, 2 * n_d), max(5, 0.75 * n_t)))
    ax.imshow(matrix, cmap=plt.cm.RdYlGn, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_d))
    ax.set_xticklabels([r.label for r in reports],
                       fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_t))
    ax.set_yticklabels([t.name.replace("_", "\n")
                       for t, _ in all_tests], fontsize=8.5)
    for i in range(n_t):
        for j in range(n_d):
            ax.text(j, i, "PASS" if matrix[i, j] > 0.5 else "FAIL",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white")
    n_san = len(reports[0].sanity)
    ax.axhline(n_san - 0.5, color="white", linewidth=2.5)
    ax.set_title("Test Results", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "test_results")


def plot_test_metrics(reports, sigma_lo, sigma_hi, out_dir):
    n = len(reports[0].sensitivity)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        axes = [axes]
    for col in range(n):
        ax = axes[col]
        t0 = reports[0].sensitivity[col]
        metrics = [r.sensitivity[col].metric for r in reports]
        passed = [r.sensitivity[col].passed for r in reports]
        colours = [PALETTE[r.label] for r in reports]
        bars = ax.bar(range(len(reports)), metrics, color=colours)
        for bar, p in zip(bars, passed):
            bar.set_alpha(0.9 if p else 0.5)
        ax.axhline(t0.threshold, color="black", linewidth=1.5,
                   linestyle="--", label=f"threshold={t0.threshold}")
        ax.set_xticks(range(len(reports)))
        ax.set_xticklabels([r.label for r in reports],
                           rotation=25, ha="right", fontsize=9)
        ax.set_title(t0.name.replace("_", " "), fontsize=10)
        if col == 0:
            ax.set_ylabel(f"Metric ({t0.unit})" if t0.unit else "Metric")
        ax.legend(fontsize=8, frameon=False)
        ylo, yhi = ax.get_ylim()
        for i, (m, p) in enumerate(zip(metrics, passed)):
            ax.text(i, m + 0.01 * (yhi - ylo), "✓" if p else "✗",
                    ha="center", va="bottom", fontsize=13,
                    color="#2ca02c" if p else "#d62728")
    fig.suptitle(f"Signal sensitivity metrics  (σ={sigma_lo} vs σ={sigma_hi})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "test_metrics")


def generate_all_plots(reports, sigma_data_all, sigmas, out_dir):
    plt.rcParams.update(STYLE)
    sigma_lo, sigma_hi = sigmas[0], sigmas[-1]
    plot_mcspu_vs_sigma(reports, sigma_data_all, sigmas, out_dir)
    plot_distributions(reports, sigma_data_all, sigmas, out_dir)
    plot_heatmap(reports, sigma_data_all, sigmas, out_dir)
    plot_test_results(reports, out_dir)
    plot_test_metrics(reports, sigma_lo, sigma_hi, out_dir)


# MAIN

def parse_args():
    p = argparse.ArgumentParser(
        description="OpenTSLM end-to-end MCSPU uncertainty test suite (single model)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    #  model
    p.add_argument("--checkpoint",  type=Path, required=True,
                   help="Path to the .pt checkpoint file")
    p.add_argument("--dataset",     required=True, choices=list(STAGE_REGISTRY.keys()),
                   help="Dataset key: tsqa | har | sleep | ecg_qa")
    p.add_argument("--model_type",  default="sp", choices=["sp", "flamingo"],
                   help="Model architecture")
    p.add_argument("--llm_id",      default="meta-llama/Llama-3.2-1B",
                   help="HuggingFace LLM model ID")
    p.add_argument("--device",      default=None,
                   help="cuda | cpu | mps (default: auto-detect)")
    #  scoring
    p.add_argument("--n_noise",     type=int,   default=50,
                   help="Noise draws N per sample")
    p.add_argument("--max_samples", type=int,   default=200,
                   help="Max test samples per sigma")
    p.add_argument("--class_batch", type=int,   default=8,
                   help="Max answer candidates per GPU forward pass")
    p.add_argument("--sigmas",      type=float, nargs="+",
                   default=[0.1, 0.5, 1.0, 2.0],
                   help="Noise levels to sweep")
    p.add_argument("--seed",        type=int,   default=42)
    #  output
    p.add_argument("--out_dir",     type=Path, default=Path("plots"),
                   help="Output directory for plots")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sigmas = sorted(args.sigmas)
    sigma_lo, sigma_hi = sigmas[0], sigmas[-1]

    cfg = STAGE_REGISTRY[args.dataset]

    print(f"\nOpenTSLM MCSPU Production Gate")
    print(f"  checkpoint={args.checkpoint}")
    print(f"  dataset={args.dataset}  model_type={args.model_type}")
    print(f"  device={device}  n_noise={args.n_noise}  max_samples={args.max_samples}"
          f"  class_batch={args.class_batch}  sigmas={sigmas}\n")

    if not args.checkpoint.exists():
        print(f"[error] checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model = load_model(args.checkpoint, args.llm_id, device, args.model_type)
    dataset = load_dataset_split(args.dataset, model.get_eos_token())
    print(f"  dataset size: {len(dataset)} test samples", flush=True)

    sigma_data: Dict[float, List[dict]] = {}
    for sigma in sigmas:
        print(f"  σ={sigma}", flush=True)
        sigma_data[sigma] = score_dataset(
            model, dataset,
            default_vocab=cfg["answer_vocab"],
            n_noise=args.n_noise,
            sigma=sigma,
            class_batch=args.class_batch,
            max_samples=args.max_samples,
            seed=args.seed,
        )

    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    report = StageReport(
        stage_key=args.dataset,
        label=cfg["label"],
        sanity=run_sanity(sigma_data),
        sensitivity=run_sensitivity(sigma_data, sigma_lo, sigma_hi),
    )
    reports = [report]
    sigma_data_all = {args.dataset: sigma_data}

    exit_code = print_report(reports)

    print("Generating plots …")
    generate_all_plots(reports, sigma_data_all, sigmas, args.out_dir)
    print(f"Plots saved to {args.out_dir}/\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
