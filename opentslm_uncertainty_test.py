#!/usr/bin/env python3
"""
opentslm_uncertainty_test.py — End-to-end MCSPU uncertainty test suite.

Loads each OpenTSLM stage checkpoint, scores it with Monte Carlo Signal
Perturbation Uncertainty (MCSPU) across four noise levels, runs sanity and
signal-sensitivity tests, and writes a colour-coded report + plots.

Your team only needs to run this one script.  All MCSPU computation is
implemented here; the only external dependency is the OpenTSLM src/ package
(for model and dataset classes).

Usage
-----
    python opentslm_uncertainty_test.py                      # all defaults
    python opentslm_uncertainty_test.py \\
        --models_dir  models/ \\
        --llm_id      meta-llama/Llama-3.2-1B \\
        --device      cuda \\
        --n_noise     50 \\
        --max_samples 200 \\
        --class_batch 8 \\
        --sigmas      0.1 0.5 1.0 2.0 \\
        --out_dir     plots/ \\
        --skip        ecg_qa          # skip one stage (comma-separated)

Exit code
---------
    0  all outcomes match expectations
    1  unexpected pass or fail detected
"""

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
import matplotlib.pyplot as plt

# ── make src/ importable regardless of working directory ────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))
sys.path.insert(0, str(_HERE / "src" / "open_flamingo"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ============================================================================
# 1. STAGE REGISTRY
#    Edit checkpoint filenames here if your paths differ.
# ============================================================================

STAGE_REGISTRY = {
    "tsqa": {
        "label":               "TSQA",
        "checkpoint":          "stage1_tsqa_sp.pt",
        "dataset":             "tsqa",
        "answer_vocab":        ["A", "B", "C", "D"],
        "expected_uses_signal": True,
    },
    "har": {
        "label":               "HAR",
        "checkpoint":          "stage3_har_sp.pt",
        "dataset":             "har",
        "answer_vocab":        [
            "biking", "lying", "running", "sitting",
            "standing", "walking", "walking_down", "walking_up",
        ],
        "expected_uses_signal": True,
    },
    "sleep": {
        "label":               "Sleep EDF",
        "checkpoint":          "stage4_sleep_sp.pt",
        "dataset":             "sleep",
        "answer_vocab":        [
            "Wake", "Non-REM stage 1", "Non-REM stage 2",
            "Non-REM stage 3", "REM sleep", "Movement",
        ],
        "expected_uses_signal": True,
    },
    "ecg_qa": {
        "label":               "ECG-QA",
        "checkpoint":          "stage5_ecg_sp.pt",
        "dataset":             "ecg_qa",
        "answer_vocab":        None,     # per-sample vocab from dataset
        "expected_uses_signal": False,
    },
}

# ============================================================================
# 2. TEST THRESHOLDS
#    See UNCERTAINTY_TEST_GUIDE.txt for full derivation from experimental data.
# ============================================================================

THRESHOLDS = {
    "min_mcspu_high_sigma": 0.05,   # mean MCSPU at σ_max  (nats)
    "min_mcspu_range":      0.02,   # MCSPU(σ_max) − MCSPU(σ_min)
    "max_mannwhitney_p":    0.05,   # one-sided p-value
    "min_cohens_d":         0.30,   # Cohen's d
    "kl_consistency_atol":  1e-5,   # |score − mean(kl)| per record
    "prob_sum_atol":        1e-3,   # |Σ probs − 1|
}


# ============================================================================
# 3. MCSPU COMPUTATION  (self-contained — no import from mcspu.py)
# ============================================================================

def _add_noise(sample: Dict[str, Any], sigma: float, rng: np.random.Generator) -> Dict[str, Any]:
    """Deep-copy sample and add Gaussian noise to time_series only."""
    noisy = copy.deepcopy(sample)
    ts = noisy["time_series"]
    if isinstance(ts, torch.Tensor):
        noise = torch.tensor(rng.normal(0.0, sigma, ts.shape), dtype=ts.dtype, device=ts.device)
        noisy["time_series"] = ts + noise
    else:  # List[List[float]]
        noisy["time_series"] = [
            (np.asarray(ch) + rng.normal(0.0, sigma, len(ch))).tolist()
            for ch in ts
        ]
    return noisy


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    p, q = p.float(), q.float()
    return float(torch.sum(p * (torch.log(p + eps) - torch.log(q + eps))).item())


def _get_logprobs(model, sample: Dict[str, Any], vocab: List[str],
                  class_batch: int) -> torch.Tensor:
    """
    Collate *sample* and call model.compute_class_logprobs with chunked batching.
    Returns a 1-D CPU float32 tensor of shape (len(vocab),).
    """
    from opentslm.time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate,
    )
    collated = extend_time_series_to_match_patch_size_and_aggregate([copy.deepcopy(sample)])[0]
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
    clean_lp  = _get_logprobs(model, sample, vocab, class_batch)
    p0        = torch.softmax(clean_lp.float(), dim=0)

    per_kl: List[float] = []
    for _ in range(n_noise):
        noisy    = _add_noise(sample, sigma, rng)
        noisy_lp = _get_logprobs(model, noisy, vocab, class_batch)
        pi       = torch.softmax(noisy_lp.float(), dim=0)
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
    n   = min(max_samples, len(dataset))
    results = []

    for idx in range(n):
        sample = dataset[idx]
        vocab  = sample.get("possible_answers") or default_vocab
        result = score_sample(model, sample, vocab, n_noise, sigma, class_batch, rng)
        result["sample_idx"]   = idx
        result["ground_truth"] = sample.get("answer", "")
        results.append(result)

        if (idx + 1) % 10 == 0 or (idx + 1) == n:
            print(f"    [{idx+1:>4}/{n}]  mcspu={result['mcspu_score']:.4f}"
                  f"  pred={result['clean_pred']!r}", flush=True)

    return results


# ============================================================================
# 4. MODEL & DATASET LOADING
# ============================================================================

def load_model(checkpoint: Path, llm_id: str, device: str):
    from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
    print(f"  Loading checkpoint {checkpoint.name} on {device} …", flush=True)
    model = OpenTSLMSP(llm_id=llm_id, device=device)
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


# ============================================================================
# 5. TEST SUITE
# ============================================================================

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
    expected_uses_signal: bool
    sanity: List[TestResult]      = field(default_factory=list)
    sensitivity: List[TestResult] = field(default_factory=list)

    def sanity_ok(self)      -> bool: return all(t.passed for t in self.sanity)
    def sensitivity_ok(self) -> bool: return all(t.passed for t in self.sensitivity)

    def outcome_as_expected(self) -> bool:
        if self.expected_uses_signal:
            return self.sanity_ok() and self.sensitivity_ok()
        return self.sanity_ok() and not self.sensitivity_ok()


def run_sanity(sigma_data: Dict[float, List[dict]]) -> List[TestResult]:
    all_r = [r for records in sigma_data.values() for r in records]
    out   = []

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
    worst = max(abs(r["mcspu_score"] - np.mean(r["per_noise_kl"])) for r in all_r)
    thr   = THRESHOLDS["kl_consistency_atol"]
    out.append(TestResult("score_consistency", worst <= thr,
                          float(worst), thr, "<=", "nats",
                          f"max |score − mean(kl)| = {worst:.2e}"))

    # 4. probs sum to 1
    worst_sum = max(abs(sum(r["clean_probs"]) - 1.0) for r in all_r)
    thr2      = THRESHOLDS["prob_sum_atol"]
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
    thr     = THRESHOLDS["min_mcspu_high_sigma"]
    out.append(TestResult("sensitivity_magnitude", mean_hi > thr,
                          mean_hi, thr, ">", "nats",
                          f"mean MCSPU(σ={sigma_hi}) = {mean_hi:.4f}"))

    # 2. absolute growth
    delta = float(hi.mean() - lo.mean())
    thr   = THRESHOLDS["min_mcspu_range"]
    out.append(TestResult("sensitivity_range", delta > thr,
                          delta, thr, ">", "nats",
                          f"MCSPU({sigma_hi}) − MCSPU({sigma_lo}) = {delta:.4f}"))

    # 3. Mann-Whitney U (one-sided)
    _, p = stats.mannwhitneyu(hi, lo, alternative="greater")
    thr  = THRESHOLDS["max_mannwhitney_p"]
    out.append(TestResult("statistical_significance", float(p) < thr,
                          float(p), thr, "<", "p-value",
                          f"Mann-Whitney U p = {p:.2e}"))

    # 4. Cohen's d
    d   = _cohens_d(hi, lo)
    thr = THRESHOLDS["min_cohens_d"]
    out.append(TestResult("effect_size", d > thr,
                          d, thr, ">", "σ",
                          f"Cohen's d = {d:.2f}"))

    return out


# ============================================================================
# 6. TERMINAL REPORT
# ============================================================================

_G, _R, _Y, _RST, _B = "\033[92m", "\033[91m", "\033[93m", "\033[0m", "\033[1m"

def _c(txt, col): return f"{col}{txt}{_RST}"
def _pass(p):     return _c("PASS", _G) if p else _c("FAIL", _R)


def print_report(reports: List[StageReport]) -> int:
    div = "═" * 72
    print(f"\n{_B}{div}{_RST}")
    print(f"{_B}  MCSPU UNCERTAINTY TEST SUITE{_RST}")
    print(f"{_B}{div}{_RST}\n")

    def _table(title, get_tests):
        tests  = get_tests(reports[0])
        names  = [t.name for t in tests]
        w      = max(len(n) for n in names) + 2
        hdr    = "  ".join(f"{r.label:>12}" for r in reports)
        print(f"{_B}{title}{_RST}")
        print(f"  {'Test':{w}s}  {hdr}")
        print("  " + "─" * (w + 2 + 14 * len(reports)))
        for i, name in enumerate(names):
            row = f"  {name:{w}s}  "
            row += "  ".join(f"{_pass(get_tests(r)[i].passed):>12}" for r in reports)
            print(row)
        print()

    _table("SANITY TESTS  (all models must pass)",
           lambda r: r.sanity)
    _table("SIGNAL SENSITIVITY TESTS  (TSQA/HAR/Sleep pass · ECG-QA fails)",
           lambda r: r.sensitivity)

    # threshold table
    print("─" * 72)
    print(f"  {'Sensitivity threshold':30s}  {'Value':>8}  Direction  Unit")
    print("─" * 72)
    for t in reports[0].sensitivity:
        print(f"  {t.name:<30}  {t.threshold:>8.4f}  {t.op:<9}  {t.unit}")

    # metric values
    print()
    hdr = "  ".join(f"{r.label:>12}" for r in reports)
    print(f"  {'Metric (σ_lo vs σ_hi)':30s}  {hdr}")
    print("─" * 72)
    for i, t in enumerate(reports[0].sensitivity):
        vals = "  ".join(f"{r.sensitivity[i].metric:>12.4f}" for r in reports)
        print(f"  {t.name:<30}  {vals}")

    # verdicts
    print(f"\n{_B}OVERALL VERDICTS{_RST}")
    print("─" * 72)
    unexpected = 0
    for r in reports:
        signal_str = _c("USES SIGNAL",    _G) if r.sensitivity_ok() else _c("IGNORES SIGNAL", _R)
        verdict    = _c("✓  AS EXPECTED", _G) if r.outcome_as_expected() else _c("✗  UNEXPECTED", _R)
        if not r.outcome_as_expected():
            unexpected += 1
        print(f"  {r.label:<12}  {signal_str:<35}  {verdict}")

    print()
    msg = "All outcomes match expectations." if unexpected == 0 else f"{unexpected} unexpected outcome(s)."
    col = _G if unexpected == 0 else _R
    print(_c(f"  {msg}  Exit {int(unexpected > 0)}.", col))
    print()
    return int(unexpected > 0)


# ============================================================================
# 7. PLOTS
# ============================================================================

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
    for ext in (".pdf", ".png"):
        fig.savefig(str(stem) + ext, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}.pdf")


def plot_mcspu_vs_sigma(reports, sigma_data_all, sigmas, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    for rep in reports:
        sdata  = sigma_data_all[rep.stage_key]
        colour = PALETTE[rep.label]
        xs, ys, cis = [], [], []
        for s in sigmas:
            sc = np.array([r["mcspu_score"] for r in sdata.get(s, [])])
            if len(sc) == 0: continue
            xs.append(s); ys.append(sc.mean())
            cis.append(1.96 * sc.std() / np.sqrt(len(sc)))
        xs, ys, cis = np.array(xs), np.array(ys), np.array(cis)
        ls = "-" if rep.sensitivity_ok() else "--"
        ax.plot(xs, ys, marker="o", color=colour, linewidth=2, linestyle=ls,
                label=f"{rep.label}  [{'USES' if rep.sensitivity_ok() else 'IGNORES'} SIGNAL]")
        ax.fill_between(xs, ys - cis, ys + cis, alpha=0.12, color=colour)
    ax.set_xlabel("Noise σ"); ax.set_ylabel("Mean MCSPU (± 95% CI)")
    ax.set_title("Signal sensitivity vs noise magnitude")
    ax.set_xticks(sigmas); ax.legend(fontsize=9, frameon=False)
    fig.tight_layout(); _save(fig, out_dir / "mcspu_vs_sigma")


def plot_distributions(reports, sigma_data_all, sigmas, out_dir):
    fig, axes = plt.subplots(1, len(sigmas), figsize=(4 * len(sigmas), 4.5), sharey=False)
    for col, sigma in enumerate(sigmas):
        ax = axes[col]
        scores_list, positions, colours = [], [], []
        for i, rep in enumerate(reports):
            sc = [r["mcspu_score"] for r in sigma_data_all[rep.stage_key].get(sigma, [])]
            if not sc: continue
            scores_list.append(sc); positions.append(i + 1)
            colours.append(PALETTE[rep.label])
        vp = ax.violinplot(scores_list, positions=positions, showmedians=True, showextrema=False)
        for body, col_ in zip(vp["bodies"], colours):
            body.set_facecolor(col_); body.set_alpha(0.7)
        vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(1.5)
        ax.set_xticks(positions)
        ax.set_xticklabels([r.label for r in reports], rotation=30, ha="right", fontsize=9)
        ax.set_title(f"σ = {sigma}")
        if col == 0: ax.set_ylabel("MCSPU score")
    fig.suptitle("Per-sample MCSPU distributions", fontsize=12)
    fig.tight_layout(); _save(fig, out_dir / "distributions")


def plot_heatmap(reports, sigma_data_all, sigmas, out_dir):
    matrix = np.full((len(reports), len(sigmas)), np.nan)
    for i, rep in enumerate(reports):
        for j, s in enumerate(sigmas):
            sc = [r["mcspu_score"] for r in sigma_data_all[rep.stage_key].get(s, [])]
            if sc: matrix[i, j] = np.mean(sc)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Mean MCSPU")
    ax.set_xticks(range(len(sigmas))); ax.set_xticklabels([f"σ={s}" for s in sigmas])
    ax.set_yticks(range(len(reports))); ax.set_yticklabels([r.label for r in reports])
    ax.set_title("Mean MCSPU (dataset × σ)")
    vmid = np.nanmean(matrix)
    for i in range(len(reports)):
        for j in range(len(sigmas)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                        fontweight="bold", color="white" if v > vmid else "black")
    fig.tight_layout(); _save(fig, out_dir / "heatmap")


def plot_test_results(reports, out_dir):
    all_tests = ([(t, "sanity") for t in reports[0].sanity] +
                 [(t, "sensitivity") for t in reports[0].sensitivity])
    n_t, n_d  = len(all_tests), len(reports)
    matrix    = np.zeros((n_t, n_d))
    for j, rep in enumerate(reports):
        for i, (_, cat) in enumerate(all_tests):
            src   = rep.sanity if cat == "sanity" else rep.sensitivity
            idx_  = i if cat == "sanity" else i - len(rep.sanity)
            matrix[i, j] = 1.0 if src[idx_].passed else 0.0

    fig, ax = plt.subplots(figsize=(max(6, 2 * n_d), max(5, 0.75 * n_t)))
    ax.imshow(matrix, cmap=plt.cm.RdYlGn, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_d)); ax.set_xticklabels([r.label for r in reports],
                                                   fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_t))
    ax.set_yticklabels([t.name.replace("_", "\n") for t, _ in all_tests], fontsize=8.5)
    for i in range(n_t):
        for j in range(n_d):
            ax.text(j, i, "PASS" if matrix[i, j] > 0.5 else "FAIL",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white")
    n_san = len(reports[0].sanity)
    ax.axhline(n_san - 0.5, color="white", linewidth=2.5)
    ax.set_title("Test Results", fontsize=13, fontweight="bold")
    fig.tight_layout(); _save(fig, out_dir / "test_results")


def plot_test_metrics(reports, sigma_lo, sigma_hi, out_dir):
    n   = len(reports[0].sensitivity)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1: axes = [axes]
    for col in range(n):
        ax  = axes[col]
        t0  = reports[0].sensitivity[col]
        metrics = [r.sensitivity[col].metric for r in reports]
        passed  = [r.sensitivity[col].passed for r in reports]
        colours = [PALETTE[r.label] for r in reports]
        ax.bar(range(len(reports)), metrics, color=colours,
               alpha=[0.9 if p else 0.5 for p in passed])
        ax.axhline(t0.threshold, color="black", linewidth=1.5,
                   linestyle="--", label=f"threshold={t0.threshold}")
        ax.set_xticks(range(len(reports)))
        ax.set_xticklabels([r.label for r in reports], rotation=25, ha="right", fontsize=9)
        ax.set_title(t0.name.replace("_", " "), fontsize=10)
        if col == 0: ax.set_ylabel(f"Metric ({t0.unit})" if t0.unit else "Metric")
        ax.legend(fontsize=8, frameon=False)
        ylo, yhi = ax.get_ylim()
        for i, (m, p) in enumerate(zip(metrics, passed)):
            ax.text(i, m + 0.01 * (yhi - ylo), "✓" if p else "✗",
                    ha="center", va="bottom", fontsize=13,
                    color="#2ca02c" if p else "#d62728")
    fig.suptitle(f"Signal sensitivity metrics  (σ={sigma_lo} vs σ={sigma_hi})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); _save(fig, out_dir / "test_metrics")


def generate_all_plots(reports, sigma_data_all, sigmas, out_dir):
    plt.rcParams.update(STYLE)
    sigma_lo, sigma_hi = sigmas[0], sigmas[-1]
    plot_mcspu_vs_sigma(reports, sigma_data_all, sigmas, out_dir)
    plot_distributions(reports, sigma_data_all, sigmas, out_dir)
    plot_heatmap(reports, sigma_data_all, sigmas, out_dir)
    plot_test_results(reports, out_dir)
    plot_test_metrics(reports, sigma_lo, sigma_hi, out_dir)


# ============================================================================
# 8. MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="OpenTSLM end-to-end MCSPU uncertainty test suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── paths ──────────────────────────────────────────────────────────────
    p.add_argument("--models_dir",  type=Path, default=Path("models"),
                   help="Directory containing .pt checkpoint files")
    p.add_argument("--out_dir",     type=Path, default=Path("plots"),
                   help="Output directory for plots")
    # ── model ──────────────────────────────────────────────────────────────
    p.add_argument("--llm_id",      default="meta-llama/Llama-3.2-1B",
                   help="HuggingFace LLM model ID")
    p.add_argument("--device",      default=None,
                   help="cuda | cpu | mps (default: auto-detect)")
    # ── scoring ────────────────────────────────────────────────────────────
    p.add_argument("--n_noise",     type=int,   default=50,
                   help="Noise draws N per sample")
    p.add_argument("--max_samples", type=int,   default=200,
                   help="Max test samples per stage/sigma")
    p.add_argument("--class_batch", type=int,   default=8,
                   help="Max answer candidates per GPU forward pass")
    p.add_argument("--sigmas",      type=float, nargs="+",
                   default=[0.1, 0.5, 1.0, 2.0],
                   help="Noise levels to sweep")
    p.add_argument("--seed",        type=int,   default=42)
    # ── control ────────────────────────────────────────────────────────────
    p.add_argument("--skip",        default="",
                   help="Comma-separated stage keys to skip (e.g. ecg_qa,har)")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device  = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    skip    = {s.strip() for s in args.skip.split(",") if s.strip()}
    sigmas  = sorted(args.sigmas)
    sigma_lo, sigma_hi = sigmas[0], sigmas[-1]

    print(f"\nOpenTSLM Uncertainty Test Suite")
    print(f"  device={device}  n_noise={args.n_noise}  max_samples={args.max_samples}"
          f"  class_batch={args.class_batch}  sigmas={sigmas}\n")

    sigma_data_all: Dict[str, Dict[float, List[dict]]] = {}
    reports: List[StageReport] = []

    for key, cfg in STAGE_REGISTRY.items():
        if key in skip:
            print(f"[skip] {cfg['label']}")
            continue

        ckpt = args.models_dir / cfg["checkpoint"]
        if not ckpt.exists():
            print(f"[warn] checkpoint not found: {ckpt} — skipping {cfg['label']}")
            continue

        print(f"{'─'*60}")
        print(f"Stage: {cfg['label']}  ({ckpt.name})")
        model   = load_model(ckpt, args.llm_id, device)
        dataset = load_dataset_split(cfg["dataset"], model.get_eos_token())
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

        # free GPU memory before loading next model
        del model
        torch.cuda.empty_cache() if device != "cpu" else None

        sigma_data_all[key] = sigma_data
        reports.append(StageReport(
            stage_key=key,
            label=cfg["label"],
            expected_uses_signal=cfg["expected_uses_signal"],
            sanity=run_sanity(sigma_data),
            sensitivity=run_sensitivity(sigma_data, sigma_lo, sigma_hi),
        ))

    if not reports:
        print("No stages scored — nothing to report.")
        sys.exit(1)

    exit_code = print_report(reports)

    print("Generating plots …")
    generate_all_plots(reports, sigma_data_all, sigmas, args.out_dir)
    print(f"Plots saved to {args.out_dir}/\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
