#!/usr/bin/env python3
"""
opentslm_uncertainty_test.py - MCSPU tests for a new model checkpoint

Scores one OpenTSLM checkpoint across four noise levels and decides whether the model is production-ready: it must pass all sanity tests and demonstrate that the model relies on the signal.
Apart from gaussian mode, other missing data perturbations have been added (missing_zeros, which masks with zeros certain fraction of the signal and missing_channels, dropout completely channels).
These missing data modes are only exploratory, there are no thresholds to pass there.

IMPORTANT: run with n_noise = 50 and max_samples = 200 (default), it can be changed but these values are the statistically most significant designed for this test.

Command
-----
    python opentslm_uncertainty_test.py \\
        --checkpoint  models/my_ecg_model.pt \\
        --dataset     ecg_qa \\
        --model_type  sp \\
        --llm_id      meta-llama/Llama-3.2-1B \\
        --device      cuda \\
        --n_noise     50 \\
        --max_samples 200 \\
        --class_batch 8 \\
        --perturbation_type gaussian \\ (default is gaussian, use missing_zeros or missing_channels for further analysis)
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

# Presets for missing_channels mode depending on the dataset
_CHANNEL_META = {
    "har": {
        "names": ["X", "Y", "Z"],
        "presets": [([0], "drop X"), ([1], "drop Y"), ([2], "drop Z")],
    },
    "ecg_qa": {
        "names": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "presets": [
            ([0],              "drop I"),
            ([1],              "drop II"),
            ([6],              "drop V1"),
            ([0, 1, 2],        "drop std. limb (I,II,III)"),
            ([0, 1, 2, 3, 4, 5], "drop all limb"),
            ([6, 7, 8, 9, 10, 11], "drop precordial (V1-V6)"),
        ],
    },
    "tsqa":  {"names": ["ch0"], "presets": [([0], "drop ch0 (blackout)")]},
    "sleep": {"names": ["EEG"], "presets": [([0], "drop ch0 (blackout)")]},
}


def _auto_channel_configs(dataset_name: str) -> List[Tuple[List[int], str]]:
    """Return (channel_indices, label) presets for missing_channels auto mode."""
    meta = _CHANNEL_META.get(dataset_name)
    if meta is None:
        return [([0], "drop ch0")]
    return meta["presets"]


def _channel_config_label(channels: List[int], dataset_name: str) -> str:
    """Human-readable label for a custom channel config."""
    names = (_CHANNEL_META.get(dataset_name) or {}).get("names", [])
    parts = [names[c] if c < len(names) else str(c) for c in channels]
    return "drop " + "+".join(parts)


# TEST THRESHOLDS: See UNCERTAINTY_TEST_GUIDE.txt for full derivation from experimental data.
# SP and Flamingo have separate magnitude/range thresholds because Flamingo's cross-attention
# gates remain near-zero early in training, structurally reducing absolute KL by ~10-20x vs SP.
# Statistical tests (p-value, Cohen's d) are architecture-neutral and identical in both sets.

SP_THRESHOLDS = {
    "min_mcspu_high_sigma": 0.05,   # mean MCSPU at σ_max  (nats)
    "min_mcspu_range":      0.02,   # MCSPU(σ_max) − MCSPU(σ_min)
    "max_mannwhitney_p":    0.05,   # one-sided p-value
    "min_cohens_d":         0.30,   # Cohen's d
    "kl_consistency_atol":  1e-5,   # |score − mean(kl)| per record
    "prob_sum_atol":        1e-3,   # |Σ probs − 1|
}

# Derived from: har_flamingo (0.0570, PASS), sleep_flamingo (0.0103, PASS), ecg_flamingo (0.0004, FAIL).
# Threshold of 0.008 sits 1.3x below sleep (tightest passing) and 20x above ECG (failing).
FLAMINGO_THRESHOLDS = {
    "min_mcspu_high_sigma": 0.008,  # lower than SP: Flamingo gates suppress absolute KL
    "min_mcspu_range":      0.008,  # lower than SP: same reason
    "max_mannwhitney_p":    0.05,   # unchanged: architecture-neutral
    "min_cohens_d":         0.30,   # unchanged: architecture-neutral
    "kl_consistency_atol":  1e-5,
    "prob_sum_atol":        1e-3,
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


def _mask_missing(sample: Dict[str, Any], fraction: float, rng: np.random.Generator) -> Dict[str, Any]:
    """Deep-copy sample and zero out a random fraction of timepoints per channel."""
    from opentslm.uncertainty.mcspu import add_missing_data
    return add_missing_data(sample, fraction, rng)


def _mask_channel_dropout(sample: Dict[str, Any], channels_to_drop: List[int], rng: np.random.Generator) -> Dict[str, Any]:
    """Deep-copy sample and zero out the specified channels entirely."""
    from opentslm.uncertainty.mcspu import add_channel_dropout
    return add_channel_dropout(sample, channels_to_drop, rng)


def _kl(lp: torch.Tensor, lq: torch.Tensor) -> float:
    """KL(p||q) from raw sequence log-prob vectors, computed in float64.

    Accepts the unnormalised logit vectors returned by compute_class_logprobs
    and applies log-softmax internally.  Using float64 avoids the float32
    softmax collapse that turns highly-peaked distributions into exact point
    masses, which would make KL identically zero even when the logits differ.
    """
    lp = lp.double()
    lq = lq.double()
    log_p = lp - torch.logsumexp(lp, dim=0)   # numerically stable log-softmax
    log_q = lq - torch.logsumexp(lq, dim=0)
    p = log_p.exp()                             # probability vector in float64
    return float(max(0.0, torch.sum(p * (log_p - log_q)).item()))


def _get_logprobs(model, sample: Dict[str, Any], vocab: List[str],
                  class_batch: int) -> torch.Tensor:
    """
    Collate sample and call model.compute_class_logprobs with chunked batching.
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
    perturb_fn,
    class_batch: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Run N+1 forward passes and return the MCSPU record for one sample.
    perturb_fn(sample, rng) -> perturbed_sample
    """
    # clean pass
    clean_lp = _get_logprobs(model, sample, vocab, class_batch)
    p0 = torch.softmax(clean_lp.float(), dim=0)

    per_kl: List[float] = []
    for _ in range(n_noise):
        perturbed = perturb_fn(sample, rng)
        perturbed_lp = _get_logprobs(model, perturbed, vocab, class_batch)
        per_kl.append(_kl(clean_lp, perturbed_lp))
        if model.device != "cpu":
            torch.cuda.empty_cache()

    return {
        "mcspu_score":    float(np.mean(per_kl)),
        "clean_logprobs": clean_lp.tolist(),
        "clean_probs":    p0.tolist(),
        "clean_pred":     vocab[int(torch.argmax(p0).item())],
        "per_noise_kl":   per_kl,
        "n_samples":      n_noise,
        "answer_vocab":   vocab,
    }


def score_dataset(
    model,
    dataset,
    default_vocab: Optional[List[str]],
    n_noise: int,
    perturb_fn,
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
                              n_noise, perturb_fn, class_batch, rng)
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


def run_sanity(sigma_data: Dict[float, List[dict]], thresholds: dict) -> List[TestResult]:
    all_r = [r for records in sigma_data.values() for r in records]
    out = []

    # 1. no NaN/Inf
    bad = sum(1 for r in all_r if not np.isfinite(r["mcspu_score"]))
    out.append(TestResult("no_nan_inf_scores", bad == 0,
                          float(bad), 0, "==", "bad records",
                          f"{bad}/{len(all_r)} non-finite scores"))

    # 2. KL ≥ 0
    min_kl = min(kl for r in all_r for kl in r["per_noise_kl"])
    out.append(TestResult("kl_nonneg", min_kl >= -1e-6,
                          float(min_kl), 0.0, ">=", "nats",
                          f"min per_noise_kl = {min_kl:.2e}"))

    # 3. score == mean(kl)
    worst = max(abs(r["mcspu_score"] - np.mean(r["per_noise_kl"]))
                for r in all_r)
    thr = thresholds["kl_consistency_atol"]
    out.append(TestResult("score_consistency", worst <= thr,
                          float(worst), thr, "<=", "nats",
                          f"max |score − mean(kl)| = {worst:.2e}"))

    # 4. probs sum to 1
    worst_sum = max(abs(sum(r["clean_probs"]) - 1.0) for r in all_r)
    thr2 = thresholds["prob_sum_atol"]
    out.append(TestResult("probs_normalized", worst_sum <= thr2,
                          float(worst_sum), thr2, "<=", "",
                          f"max |Σ probs − 1| = {worst_sum:.2e}"))
    return out


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


def run_sensitivity(sigma_data: Dict[float, List[dict]],
                    sigma_lo: float, sigma_hi: float,
                    thresholds: dict) -> List[TestResult]:
    lo = np.array([r["mcspu_score"] for r in sigma_data.get(sigma_lo, [])])
    hi = np.array([r["mcspu_score"] for r in sigma_data.get(sigma_hi, [])])
    if len(lo) == 0 or len(hi) == 0:
        return []
    out = []

    # 1. magnitude at high sigma
    mean_hi = float(hi.mean())
    thr = thresholds["min_mcspu_high_sigma"]
    out.append(TestResult("sensitivity_magnitude", mean_hi > thr,
                          mean_hi, thr, ">", "nats",
                          f"mean MCSPU(σ={sigma_hi}) = {mean_hi:.4f}"))

    # 2. absolute growth
    delta = float(hi.mean() - lo.mean())
    thr = thresholds["min_mcspu_range"]
    out.append(TestResult("sensitivity_range", delta > thr,
                          delta, thr, ">", "nats",
                          f"MCSPU({sigma_hi}) − MCSPU({sigma_lo}) = {delta:.4f}"))

    # 3. Mann-Whitney U (one-sided)
    _, p = stats.mannwhitneyu(hi, lo, alternative="greater")
    thr = thresholds["max_mannwhitney_p"]
    out.append(TestResult("statistical_significance", float(p) < thr,
                          float(p), thr, "<", "p-value",
                          f"Mann-Whitney U p = {p:.2e}"))

    # 4. Cohen's d
    d = _cohens_d(hi, lo)
    thr = thresholds["min_cohens_d"]
    out.append(TestResult("effect_size", d > thr,
                          d, thr, ">", "σ",
                          f"Cohen's d = {d:.2f}"))

    return out


# TERMINAL REPORT

_G, _R, _Y, _RST, _B = "\033[92m", "\033[91m", "\033[93m", "\033[0m", "\033[1m"


def _c(txt, col): return f"{col}{txt}{_RST}"
def _pass(p): return _c("PASS", _G) if p else _c("FAIL", _R)


def print_report(reports: List[StageReport], thresholds: dict) -> int:
    rep = reports[0]
    arch_label = "FLAMINGO" if thresholds is FLAMINGO_THRESHOLDS else "SP"
    div = "═" * 60
    print(f"\n{_B}{div}{_RST}")
    print(f"{_B}  MCSPU PRODUCTION GATE — {rep.label}{_RST}")
    print(f"{_B}{div}{_RST}\n")
    print(f"  Threshold calibration: {arch_label}")

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


def generate_all_plots(reports, sigma_data_all, sigmas, out_dir, dataset=None):
    plt.rcParams.update(STYLE)
    sigma_lo, sigma_hi = sigmas[0], sigmas[-1]
    plot_mcspu_vs_sigma(reports, sigma_data_all, sigmas, out_dir)
    plot_distributions(reports, sigma_data_all, sigmas, out_dir)
    plot_heatmap(reports, sigma_data_all, sigmas, out_dir)
    plot_test_results(reports, out_dir)
    plot_test_metrics(reports, sigma_lo, sigma_hi, out_dir)
    if dataset is not None:
        plot_signal_example_noise(dataset, sigmas, out_dir)


# MISSING DATA ANALYSIS (just exploratory, no thresholds here)


def print_missing_summary(fraction_data: Dict[float, List[dict]], label: str) -> None:
    fractions = sorted(fraction_data.keys())
    div = "─" * 52
    print(f"\n{_B}MISSING DATA ANALYSIS — {label}{_RST}")
    print(f"  {_Y}(exploratory — no pass/fail thresholds){_RST}")
    print(div)
    print(f"  {'fraction':>10}  {'mean_mcspu':>12}  {'std':>10}  {'n':>6}")
    print(div)
    for frac in fractions:
        scores = np.array([r["mcspu_score"] for r in fraction_data.get(frac, [])])
        if len(scores) == 0:
            continue
        print(f"  {frac:>10.2f}  {scores.mean():>12.6f}  {scores.std():>10.6f}  {len(scores):>6}")
    print()


def plot_mcspu_vs_fraction(label: str, fraction_data: Dict[float, List[dict]],
                           colour: str, out_dir: Path) -> None:
    fracs = sorted(fraction_data.keys())
    ys, cis = [], []
    for f in fracs:
        sc = np.array([r["mcspu_score"] for r in fraction_data.get(f, [])])
        ys.append(sc.mean())
        cis.append(1.96 * sc.std() / np.sqrt(len(sc)))
    ys, cis = np.array(ys), np.array(cis)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fracs, ys, marker="o", color=colour, linewidth=2, label=label)
    ax.fill_between(fracs, ys - cis, ys + cis, alpha=0.15, color=colour)
    ax.set_xlabel("Missing fraction")
    ax.set_ylabel("Mean MCSPU (± 95 % CI)")
    ax.set_title(f"Signal sensitivity vs missing data — {label}")
    ax.set_xticks(fracs)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    _save(fig, out_dir / "mcspu_vs_fraction")


def plot_distributions_missing(label: str, fraction_data: Dict[float, List[dict]],
                               colour: str, out_dir: Path) -> None:
    fracs = sorted(fraction_data.keys())
    fig, axes = plt.subplots(1, len(fracs), figsize=(3.5 * len(fracs), 4.5), sharey=False)
    if len(fracs) == 1:
        axes = [axes]

    for col, frac in enumerate(fracs):
        ax = axes[col]
        sc = [r["mcspu_score"] for r in fraction_data.get(frac, [])]
        if not sc:
            ax.set_title(f"fraction={frac}\n(no data)")
            continue
        vp = ax.violinplot([sc], positions=[1], showmedians=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(colour)
            body.set_alpha(0.7)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)
        ax.set_xticks([1])
        ax.set_xticklabels([label], fontsize=9)
        ax.set_title(f"fraction = {frac}")
        if col == 0:
            ax.set_ylabel("MCSPU score")

    fig.suptitle(f"Per-sample MCSPU distributions (missing data) — {label}", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir / "distributions_missing")


def plot_signal_example(dataset, fractions: List[float], out_dir: Path) -> None:
    from opentslm.uncertainty.mcspu import add_missing_data
    sample = dataset[0]
    ts = sample["time_series"]
    if isinstance(ts, torch.Tensor):
        channel = ts[0].cpu().numpy()
    else:
        channel = np.asarray(ts[0], dtype=np.float64)

    rng = np.random.default_rng(0)
    n_cols = len(fractions) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3), sharey=True)
    if n_cols == 1:
        axes = [axes]

    axes[0].plot(channel, linewidth=0.8, color="#4C72B0")
    axes[0].set_title("Original")
    axes[0].set_xlabel("Timepoint")
    axes[0].set_ylabel("Amplitude")

    for col, frac in enumerate(fractions, start=1):
        masked = add_missing_data(sample, frac, rng)
        mts = masked["time_series"]
        m_ch = mts[0].cpu().numpy() if isinstance(mts, torch.Tensor) else np.asarray(mts[0], dtype=np.float64)
        axes[col].plot(m_ch, linewidth=0.8, color="#DD8452")
        axes[col].set_title(f"Missing {int(frac * 100)}%")
        axes[col].set_xlabel("Timepoint")

    fig.suptitle("Effect of missing data masking (channel 0, sample 0)", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir / "signal_example_missing")


def plot_signal_example_noise(dataset, sigmas: List[float], out_dir: Path) -> None:
    """Show channel 0 from sample 0 with additive Gaussian noise at each sigma level."""
    sample = dataset[0]
    ts = sample["time_series"]
    if isinstance(ts, torch.Tensor):
        channel = ts[0].cpu().numpy()
    else:
        channel = np.asarray(ts[0], dtype=np.float64)

    rng = np.random.default_rng(0)
    n_cols = len(sigmas) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3), sharey=False)
    if n_cols == 1:
        axes = [axes]

    axes[0].plot(channel, linewidth=0.8, color="#4C72B0")
    axes[0].set_title("Original")
    axes[0].set_xlabel("Timepoint")
    axes[0].set_ylabel("Amplitude")

    for col, sigma in enumerate(sigmas, start=1):
        noise = rng.normal(0.0, sigma, channel.shape)
        noisy_ch = channel + noise
        axes[col].plot(channel, linewidth=0.8, color="#4C72B0", alpha=0.25)
        axes[col].plot(noisy_ch, linewidth=0.8, color="#C44E52", alpha=0.85)
        axes[col].set_title(f"σ = {sigma}")
        axes[col].set_xlabel("Timepoint")

    fig.suptitle("Effect of Gaussian noise (channel 0, sample 0)", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir / "signal_example_noise")


def plot_heatmap_missing(label: str, fraction_data: Dict[float, List[dict]],
                         out_dir: Path) -> None:
    fracs = sorted(fraction_data.keys())
    matrix = np.full((1, len(fracs)), np.nan)
    for j, f in enumerate(fracs):
        sc = [r["mcspu_score"] for r in fraction_data.get(f, [])]
        if sc:
            matrix[0, j] = np.mean(sc)

    fig, ax = plt.subplots(figsize=(max(4, 1.8 * len(fracs)), 2.5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Mean MCSPU")
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels([f"{int(f * 100)}%" for f in fracs])
    ax.set_xlabel("Missing fraction")
    ax.set_yticks([0])
    ax.set_yticklabels([label])
    ax.set_title("Mean MCSPU vs missing fraction")
    vmid = np.nanmean(matrix)
    for j in range(len(fracs)):
        v = matrix[0, j]
        if not np.isnan(v):
            ax.text(j, 0, f"{v:.4f}", ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white" if v > vmid else "black")
    fig.tight_layout()
    _save(fig, out_dir / "heatmap_missing")


# CHANNEL DROPOUT ANALYSIS (again just exploratory, no thresholds here)


def print_channel_dropout_summary(channel_data: Dict[str, List[dict]], label: str) -> None:
    div = "─" * 60
    print(f"\n{_B}CHANNEL DROPOUT ANALYSIS — {label}{_RST}")
    print(f"  {_Y}(exploratory — no pass/fail thresholds){_RST}")
    print(div)
    print(f"  {'config':30s}  {'mean_kl':>10}  {'std':>10}  {'n':>6}")
    print(div)
    for cfg_label, records in channel_data.items():
        scores = np.array([r["mcspu_score"] for r in records])
        print(f"  {cfg_label:30s}  {scores.mean():>10.6f}  {scores.std():>10.6f}  {len(scores):>6}")
    print()


def plot_mcspu_vs_channel_config(label: str, channel_data: Dict[str, List[dict]],
                                  colour: str, out_dir: Path) -> None:
    config_labels = list(channel_data.keys())
    means = [np.mean([r["mcspu_score"] for r in channel_data[lbl]]) for lbl in config_labels]
    stds  = [np.std([r["mcspu_score"]  for r in channel_data[lbl]]) for lbl in config_labels]
    ns    = [len(channel_data[lbl]) for lbl in config_labels]
    cis   = [1.96 * s / np.sqrt(n) for s, n in zip(stds, ns)]

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(config_labels)), 4))
    x = np.arange(len(config_labels))
    bars = ax.bar(x, means, color=colour, alpha=0.85, yerr=cis, capsize=4, error_kw={"elinewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean MCSPU (± 95% CI)")
    ax.set_title(f"Channel ablation sensitivity — {label}")
    offset = max(cis) * 0.05 if cis else 0.0
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{m:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / "mcspu_vs_channel_config")


def plot_distributions_channel_dropout(label: str, channel_data: Dict[str, List[dict]],
                                        colour: str, out_dir: Path) -> None:
    config_labels = list(channel_data.keys())
    n_configs = len(config_labels)
    fig, axes = plt.subplots(1, n_configs, figsize=(3.5 * n_configs, 4.5), sharey=False)
    if n_configs == 1:
        axes = [axes]

    for col, cfg_label in enumerate(config_labels):
        ax = axes[col]
        sc = [r["mcspu_score"] for r in channel_data[cfg_label]]
        if not sc:
            ax.set_title(f"{cfg_label}\n(no data)")
            continue
        vp = ax.violinplot([sc], positions=[1], showmedians=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(colour)
            body.set_alpha(0.7)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)
        ax.set_xticks([1])
        ax.set_xticklabels([cfg_label], rotation=20, ha="right", fontsize=8)
        ax.set_title(cfg_label, fontsize=9)
        if col == 0:
            ax.set_ylabel("MCSPU score")

    fig.suptitle(f"Per-sample MCSPU distributions (channel ablation) — {label}", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir / "distributions_channel_dropout")


def plot_signal_example_channel_dropout(dataset, configs: List[Tuple[List[int], str]],
                                         out_dir: Path) -> None:
    from opentslm.uncertainty.mcspu import add_channel_dropout
    sample = dataset[0]
    ts = sample["time_series"]
    if isinstance(ts, torch.Tensor):
        n_ch = ts.shape[0]
        orig = [ts[ch].cpu().numpy() for ch in range(n_ch)]
    else:
        n_ch = len(ts)
        orig = [np.asarray(ts[ch], dtype=np.float64) for ch in range(n_ch)]

    n_cols = len(configs) + 1
    fig, axes = plt.subplots(n_ch, n_cols, figsize=(3.0 * n_cols, 2.2 * n_ch), squeeze=False)

    for row in range(n_ch):
        axes[row, 0].plot(orig[row], linewidth=0.8, color="#4C72B0")
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=9)
        axes[row, 0].set_ylabel(f"ch {row}", fontsize=8)
        axes[row, 0].tick_params(labelsize=7)

    rng = np.random.default_rng(0)
    for col, (channels, cfg_label) in enumerate(configs, start=1):
        ablated = add_channel_dropout(sample, channels, rng)
        abl_ts = ablated["time_series"]
        if isinstance(abl_ts, torch.Tensor):
            abl = [abl_ts[ch].cpu().numpy() for ch in range(n_ch)]
        else:
            abl = [np.asarray(abl_ts[ch], dtype=np.float64) for ch in range(n_ch)]
        for row in range(n_ch):
            color = "#DD8452" if row in channels else "#4C72B0"
            axes[row, col].plot(abl[row], linewidth=0.8, color=color, alpha=0.85)
            if row == 0:
                axes[row, col].set_title(cfg_label, fontsize=8)
            axes[row, col].tick_params(labelsize=7)

    fig.suptitle("Channel ablation effect (sample 0)", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir / "signal_example_channel_dropout")


def generate_channel_dropout_plots(label: str, channel_data: Dict[str, List[dict]],
                                    dataset, configs: List[Tuple[List[int], str]],
                                    out_dir: Path) -> None:
    plt.rcParams.update(STYLE)
    colour = PALETTE.get(label, "#8172B2")
    plot_mcspu_vs_channel_config(label, channel_data, colour, out_dir)
    plot_distributions_channel_dropout(label, channel_data, colour, out_dir)
    plot_signal_example_channel_dropout(dataset, configs, out_dir)


def generate_missing_plots(label: str, fraction_data: Dict[float, List[dict]],
                           dataset, fractions: List[float], out_dir: Path) -> None:
    plt.rcParams.update(STYLE)
    colour = PALETTE.get(label, "#8172B2")
    plot_mcspu_vs_fraction(label, fraction_data, colour, out_dir)
    plot_distributions_missing(label, fraction_data, colour, out_dir)
    plot_signal_example(dataset, fractions, out_dir)
    plot_heatmap_missing(label, fraction_data, out_dir)


# FLAMINGO GATE DIAGNOSTIC

def check_flamingo_gates(model) -> Dict[str, float]:
    """
    Read the learned tanh(gate) values from every GatedCrossAttentionBlock.
    Returns a dict with per-layer values and a summary.
    A gate near 0.0 means the cross-attention block is closed and the model
    ignores the signal — MCSPU will be ~0 regardless of noise magnitude.
    """
    lang_encoder = model.model.lang_encoder
    if not hasattr(lang_encoder, "gated_cross_attn_layers"):
        return {"error": "no gated_cross_attn_layers found"}

    attn_gates, ff_gates = [], []
    for i, layer in enumerate(lang_encoder.gated_cross_attn_layers):
        if layer is None:
            continue
        ag = float(torch.tanh(layer.attn_gate).item())
        fg = float(torch.tanh(layer.ff_gate).item())
        attn_gates.append(ag)
        ff_gates.append(fg)

    if not attn_gates:
        return {"error": "no active gated layers"}

    return {
        "n_layers":       len(attn_gates),
        "attn_gate_mean": float(np.mean(attn_gates)),
        "attn_gate_max":  float(np.max(np.abs(attn_gates))),
        "ff_gate_mean":   float(np.mean(ff_gates)),
        "ff_gate_max":    float(np.max(np.abs(ff_gates))),
        "attn_gates":     attn_gates,
        "ff_gates":       ff_gates,
    }


def flamingo_signal_probe(model, sample: Dict[str, Any],
                          vocab: List[str], class_batch: int) -> Dict[str, Any]:
    """
    One-shot probe: score *sample* normally, then score a zeroed-signal version.
    If the two logprob vectors are identical the model is blind to the signal
    (gates closed or signal not reaching the cross-attention path).
    Returns a dict with the KL and a human-readable verdict.
    """
    from opentslm.time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate,
    )
    import copy

    def _zeroed(s):
        z = copy.deepcopy(s)
        ts = z["time_series"]
        if isinstance(ts, torch.Tensor):
            z["time_series"] = torch.zeros_like(ts)
        else:
            z["time_series"] = [[0.0] * len(ch) for ch in ts]
        return z

    collated_real = extend_time_series_to_match_patch_size_and_aggregate(
        [copy.deepcopy(sample)])[0]
    collated_zero = extend_time_series_to_match_patch_size_and_aggregate(
        [_zeroed(sample)])[0]

    with torch.no_grad():
        lp_real = model.compute_class_logprobs(collated_real, vocab, class_batch_size=class_batch)
        lp_zero = model.compute_class_logprobs(collated_zero, vocab, class_batch_size=class_batch)

    kl_val = _kl(lp_real, lp_zero)
    max_diff = float((lp_real - lp_zero).abs().max().item())
    blind = kl_val < 1e-8

    return {
        "kl_real_vs_zero": kl_val,
        "max_logprob_diff": max_diff,
        "signal_blind": blind,
    }


def run_flamingo_preflight(model, dataset, default_vocab, class_batch: int,
                           n_probe: int = 3) -> bool:
    """
    Print a gate summary and run signal-vs-zero probes.
    Returns True if the model appears to use the signal; False if gates are closed.
    """
    print(f"\n{'─'*60}")
    print("  FLAMINGO PRE-FLIGHT: GATE HEALTH CHECK")
    print(f"{'─'*60}")

    gates = check_flamingo_gates(model)
    if "error" in gates:
        print(f"  [warn] gate check failed: {gates['error']}")
    else:
        print(f"  Cross-attention layers : {gates['n_layers']}")
        print(f"  tanh(attn_gate) mean   : {gates['attn_gate_mean']:.6f}  "
              f"max|·| = {gates['attn_gate_max']:.6f}")
        print(f"  tanh(ff_gate)   mean   : {gates['ff_gate_mean']:.6f}  "
              f"max|·| = {gates['ff_gate_max']:.6f}")
        if gates["attn_gate_max"] < 1e-4:
            print(_c("  [WARN] All attention gates are effectively zero — "
                     "model will ignore the signal. MCSPU will be ~0.", _Y))

    print(f"\n  Signal-vs-zero probe ({n_probe} samples):")
    all_blind = True
    for idx in range(min(n_probe, len(dataset))):
        sample = dataset[idx]
        vocab = sample.get("possible_answers") or default_vocab
        probe = flamingo_signal_probe(model, sample, vocab, class_batch)
        status = _c("BLIND", _R) if probe["signal_blind"] else _c("SENSITIVE", _G)
        print(f"    sample {idx}: KL(real‖zero) = {probe['kl_real_vs_zero']:.2e}  "
              f"max|Δlogp| = {probe['max_logprob_diff']:.2e}  → {status}")
        if not probe["signal_blind"]:
            all_blind = False

    if all_blind:
        print(_c("\n  [FAIL] Model is BLIND to the signal on all probed samples.", _R))
        print(_c("         Gates are closed. MCSPU will be ~0 at every sigma.", _R))
        print("         Possible causes:")
        print("           • Checkpoint not trained (gates still at init=0)")
        print("           • Cross-attention layers were frozen during training")
        print("           • Model converged to ignore the time-series path")
        print(f"{'─'*60}\n")
        return False
    else:
        print(_c("\n  [OK] Model responds to signal — MCSPU should be non-zero.", _G))
        print(f"{'─'*60}\n")
        return True


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
    p.add_argument("--perturbation_type", nargs="+", default=["gaussian"],
                   choices=["gaussian", "missing_zeros", "missing_channels", "missing"],
                   help="Perturbation modes (space-separated): gaussian missing_zeros missing_channels. "
                        "'missing' is a deprecated alias for 'missing_zeros'.")
    p.add_argument("--drop_channels",
                   type=lambda s: [int(x) for x in s.split(",")],
                   action="append", default=None, dest="drop_channels",
                   metavar="INDICES",
                   help="Comma-separated channel indices to drop for missing_channels mode. "
                        "Repeat the flag for multiple configs (e.g. --drop_channels 0 --drop_channels 0,1). "
                        "If omitted, dataset-specific auto-presets are used.")
    p.add_argument("--n_noise",     type=int,   default=50,
                   help="Perturbation draws N per sample")
    p.add_argument("--max_samples", type=int,   default=200,
                   help="Max test samples per parameter value")
    p.add_argument("--class_batch", type=int,   default=8,
                   help="Max answer candidates per GPU forward pass")
    p.add_argument("--sigmas",      type=float, nargs="+",
                   default=[0.1, 0.5, 1.0, 2.0],
                   help="Noise levels to sweep (gaussian mode)")
    p.add_argument("--missing_fractions", type=float, nargs="+",
                   default=[0.1, 0.25, 0.5, 0.75, 1.0],
                   help="Fraction of timepoints to zero per channel (missing mode)")
    p.add_argument("--seed",        type=int,   default=42)
    #  output
    p.add_argument("--out_dir",     type=Path, default=Path("plots"),
                   help="Output directory for plots")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = STAGE_REGISTRY[args.dataset]

    # Normalise ptypes: map deprecated "missing" alias → "missing_zeros"
    ptypes = []
    for pt in args.perturbation_type:
        if pt == "missing":
            print(_c("[warn] --perturbation_type 'missing' is deprecated; use 'missing_zeros'.", _Y))
            ptypes.append("missing_zeros")
        else:
            ptypes.append(pt)

    print(f"\nOpenTSLM MCSPU Production Gate")
    print(f"  checkpoint={args.checkpoint}")
    print(f"  dataset={args.dataset}  model_type={args.model_type}")
    print(f"  device={device}  n_noise={args.n_noise}  max_samples={args.max_samples}"
          f"  class_batch={args.class_batch}  perturbation_type={ptypes}")
    if "gaussian" in ptypes:
        print(f"  sigmas={sorted(args.sigmas)}")
    if "missing_zeros" in ptypes:
        print(f"  missing_fractions={sorted(args.missing_fractions)}")
    if "missing_channels" in ptypes:
        if args.drop_channels:
            configs = [(ch, _channel_config_label(ch, args.dataset)) for ch in args.drop_channels]
        else:
            configs = _auto_channel_configs(args.dataset)
        print(f"  channel_configs={[label for _, label in configs]}")
    print()

    if not args.checkpoint.exists():
        print(f"[error] checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model = load_model(args.checkpoint, args.llm_id, device, args.model_type)
    dataset = load_dataset_split(args.dataset, model.get_eos_token())
    print(f"  dataset size: {len(dataset)} test samples", flush=True)

    if args.model_type == "flamingo":
        cfg_vocab = STAGE_REGISTRY[args.dataset]["answer_vocab"]
        signal_ok = run_flamingo_preflight(
            model, dataset, cfg_vocab, args.class_batch, n_probe=3
        )
        if not signal_ok:
            print(_c("[warn] Proceeding with MCSPU scoring, but expect ~0 scores.", _Y))
            print()

    thresholds = FLAMINGO_THRESHOLDS if args.model_type == "flamingo" else SP_THRESHOLDS
    exit_code = 0

    # ── GAUSSIAN: PASS/FAIL production gate ───────────────────────────────
    if "gaussian" in ptypes:
        sigmas = sorted(args.sigmas)
        sigma_lo, sigma_hi = sigmas[0], sigmas[-1]
        sigma_data: Dict[float, List[dict]] = {}
        for sigma in sigmas:
            print(f"\n  [gaussian] σ={sigma}", flush=True)
            sigma_data[sigma] = score_dataset(
                model, dataset,
                default_vocab=cfg["answer_vocab"],
                n_noise=args.n_noise,
                perturb_fn=lambda s, r, _s=sigma: _add_noise(s, _s, r),
                class_batch=args.class_batch,
                max_samples=args.max_samples,
                seed=args.seed,
            )

        report = StageReport(
            stage_key=args.dataset,
            label=cfg["label"],
            sanity=run_sanity(sigma_data, thresholds),
            sensitivity=run_sensitivity(sigma_data, sigma_lo, sigma_hi, thresholds),
        )
        exit_code = print_report([report], thresholds)

        print("Generating gaussian plots …")
        generate_all_plots([report], {args.dataset: sigma_data}, sigmas, args.out_dir,
                           dataset=dataset)
        print(f"Gaussian plots saved to {args.out_dir}/\n")

    # ── MISSING ZEROS: temporal masking, exploratory, no thresholds ───────
    if "missing_zeros" in ptypes:
        fractions = sorted(args.missing_fractions)
        fraction_data: Dict[float, List[dict]] = {}
        for frac in fractions:
            print(f"\n  [missing_zeros] fraction={frac}", flush=True)
            fraction_data[frac] = score_dataset(
                model, dataset,
                default_vocab=cfg["answer_vocab"],
                n_noise=args.n_noise,
                perturb_fn=lambda s, r, _f=frac: _mask_missing(s, _f, r),
                class_batch=args.class_batch,
                max_samples=args.max_samples,
                seed=args.seed,
            )

        print_missing_summary(fraction_data, cfg["label"])
        print("Generating missing_zeros plots …")
        generate_missing_plots(cfg["label"], fraction_data, dataset, fractions, args.out_dir)
        print(f"Missing zeros plots saved to {args.out_dir}/\n")

    # ── MISSING CHANNELS: channel ablation, exploratory, no thresholds ────
    if "missing_channels" in ptypes:
        if args.drop_channels:
            configs = [(ch, _channel_config_label(ch, args.dataset)) for ch in args.drop_channels]
        else:
            configs = _auto_channel_configs(args.dataset)

        channel_data: Dict[str, List[dict]] = {}
        for channels, label in configs:
            print(f"\n  [missing_channels] {label}  (channels={channels})", flush=True)
            channel_data[label] = score_dataset(
                model, dataset,
                default_vocab=cfg["answer_vocab"],
                n_noise=1,
                perturb_fn=lambda s, r, _ch=channels: _mask_channel_dropout(s, _ch, r),
                class_batch=args.class_batch,
                max_samples=args.max_samples,
                seed=args.seed,
            )

        print_channel_dropout_summary(channel_data, cfg["label"])
        print("Generating missing_channels plots …")
        generate_channel_dropout_plots(cfg["label"], channel_data, dataset, configs, args.out_dir)
        print(f"Missing channels plots saved to {args.out_dir}/\n")

    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
