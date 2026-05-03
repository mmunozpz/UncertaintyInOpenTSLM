#!/usr/bin/env python3
"""
uncertainty_test.py — MCSPU Uncertainty Test Suite for OpenTSLM models.

Two categories of tests are run for every stage / dataset:

  SANITY TESTS — mathematical invariants that ALL models must pass.
    These verify the MCSPU computation is correct regardless of whether
    the model uses the signal.

  SIGNAL SENSITIVITY TESTS — behavioural tests that distinguish models
    which genuinely use their input signal from those that ignore it.
    Expected outcome:
      TSQA, HAR, Sleep EDF  →  USES SIGNAL   (all sensitivity tests pass)
      ECG-QA                →  IGNORES SIGNAL (all sensitivity tests fail)

Exit code
  0  if every model's outcome matches its expectation
  1  if any unexpected result is found

Usage:
    python uncertainty_test.py
    python uncertainty_test.py --results_dir mcspu_results --out_dir plots
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "stage1_tsqa_sp":  {"label": "TSQA",      "expected_uses_signal": True},
    "stage3_har_sp":   {"label": "HAR",        "expected_uses_signal": True},
    "stage4_sleep_sp": {"label": "Sleep EDF",  "expected_uses_signal": True},
    "stage5_ecg_sp":   {"label": "ECG-QA",    "expected_uses_signal": False},
}

SIGMAS = [0.1, 0.5, 1.0, 2.0]
SIGMA_LO, SIGMA_HI = 0.1, 2.0   # extreme sigmas used for comparison tests

# ---------------------------------------------------------------------------
# Thresholds (all principled — see inline rationale)
# ---------------------------------------------------------------------------

THRESHOLDS = {
    # Signal sensitivity — must be met by signal-using models
    # --------------------------------------------------------
    # Mean MCSPU at σ=2.0:  passing models range 0.28–0.66; ECG=0.001
    "min_mcspu_high_sigma": 0.05,

    # Absolute MCSPU growth MCSPU(σ_hi) − MCSPU(σ_lo):
    # passing models: >0.27; ECG: <0.0001
    "min_mcspu_range": 0.02,

    # Mann-Whitney U one-sided p-value: scores(σ_hi) > scores(σ_lo)
    # Standard frequentist significance threshold
    "max_mannwhitney_p": 0.05,

    # Cohen's d between σ_lo and σ_hi distributions
    # 0.3 ≈ between "small" (0.2) and "medium" (0.5) effect
    # passing models: d=1.9–3.6; ECG: d=0.01
    "min_cohens_d": 0.3,

    # Sanity — must be met by ALL models (math invariants)
    # ------------------------------------------------------
    # |mcspu_score − mean(per_noise_kl)| per record
    "kl_consistency_atol": 1e-5,

    # |sum(clean_probs) − 1.0| per record
    "prob_sum_atol": 1e-3,
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    name: str
    passed: bool
    metric: float               # the measured value
    threshold: float            # the acceptance threshold
    op: str                     # ">" or "<" — direction for passing
    unit: str = ""              # display unit (e.g. "nats", "p-value")
    details: str = ""           # optional extra info

    def symbol(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class DatasetReport:
    stage_key: str
    label: str
    expected_uses_signal: bool
    sanity: List[TestResult] = field(default_factory=list)
    sensitivity: List[TestResult] = field(default_factory=list)

    def sanity_passed(self) -> bool:
        return all(t.passed for t in self.sanity)

    def sensitivity_passed(self) -> bool:
        return all(t.passed for t in self.sensitivity)

    def outcome_as_expected(self) -> bool:
        if self.expected_uses_signal:
            return self.sanity_passed() and self.sensitivity_passed()
        else:
            # For a signal-ignoring model, ALL sensitivity tests should FAIL
            return self.sanity_passed() and not self.sensitivity_passed()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_stage(results_dir: Path, stage_key: str) -> Optional[dict]:
    """Returns {sigma: [records]} or None if data is missing."""
    stage_dir = results_dir / stage_key
    if not stage_dir.exists():
        return None
    data = {}
    for sigma in SIGMAS:
        path = stage_dir / f"sigma_{sigma}.jsonl"
        if not path.exists():
            continue
        data[sigma] = [json.loads(l)
                       for l in path.read_text().splitlines() if l.strip()]
    return data if data else None


# ---------------------------------------------------------------------------
# Sanity tests
# ---------------------------------------------------------------------------

def _run_sanity(sigma_data: dict) -> List[TestResult]:
    results = []
    all_records = [r for records in sigma_data.values() for r in records]

    # 1. No NaN / Inf in MCSPU scores
    bad = sum(1 for r in all_records
              if not np.isfinite(r["mcspu_score"]))
    results.append(TestResult(
        name="no_nan_inf_scores",
        passed=(bad == 0),
        metric=float(bad),
        threshold=0,
        op="==",
        unit="bad records",
        details=f"{bad}/{len(all_records)} records have non-finite MCSPU score",
    ))

    # 2. All per-sample KL values ≥ 0  (KL divergence is always ≥ 0)
    min_kl = min(
        kl
        for r in all_records
        for kl in r["per_noise_kl"]
    )
    results.append(TestResult(
        name="kl_nonneg",
        passed=(min_kl >= -1e-9),   # tiny tolerance for float rounding
        metric=float(min_kl),
        threshold=0.0,
        op=">=",
        unit="nats",
        details=f"minimum per_noise_kl across all records: {min_kl:.2e}",
    ))

    # 3. mcspu_score == mean(per_noise_kl)  (internal consistency)
    worst_delta = max(
        abs(r["mcspu_score"] - np.mean(r["per_noise_kl"]))
        for r in all_records
    )
    thr = THRESHOLDS["kl_consistency_atol"]
    results.append(TestResult(
        name="score_consistency",
        passed=(worst_delta <= thr),
        metric=float(worst_delta),
        threshold=thr,
        op="<=",
        unit="nats",
        details=f"max |mcspu_score − mean(per_noise_kl)|: {worst_delta:.2e}",
    ))

    # 4. Probability distributions are normalised  (sum ≈ 1)
    worst_sum_err = max(
        abs(sum(r["clean_probs"]) - 1.0)
        for r in all_records
    )
    thr = THRESHOLDS["prob_sum_atol"]
    results.append(TestResult(
        name="probs_normalized",
        passed=(worst_sum_err <= thr),
        metric=float(worst_sum_err),
        threshold=thr,
        op="<=",
        unit="",
        details=f"max |Σ clean_probs − 1|: {worst_sum_err:.2e}",
    ))

    return results


# ---------------------------------------------------------------------------
# Signal sensitivity tests
# ---------------------------------------------------------------------------

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d = (mean_a − mean_b) / pooled_std."""
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


def _run_sensitivity(sigma_data: dict) -> List[TestResult]:
    results = []

    scores_lo = np.array([r["mcspu_score"]
                         for r in sigma_data.get(SIGMA_LO, [])])
    scores_hi = np.array([r["mcspu_score"]
                         for r in sigma_data.get(SIGMA_HI, [])])

    if len(scores_lo) == 0 or len(scores_hi) == 0:
        return results

    # 1. Sensitivity magnitude — mean MCSPU at high σ must exceed threshold
    mean_hi = float(scores_hi.mean())
    thr = THRESHOLDS["min_mcspu_high_sigma"]
    results.append(TestResult(
        name="sensitivity_magnitude",
        passed=(mean_hi > thr),
        metric=mean_hi,
        threshold=thr,
        op=">",
        unit="nats",
        details=f"mean MCSPU at σ={SIGMA_HI}: {mean_hi:.4f}  (threshold: >{thr})",
    ))

    # 2. Sensitivity range — absolute growth MCSPU(σ_hi) − MCSPU(σ_lo)
    delta = float(scores_hi.mean() - scores_lo.mean())
    thr = THRESHOLDS["min_mcspu_range"]
    results.append(TestResult(
        name="sensitivity_range",
        passed=(delta > thr),
        metric=delta,
        threshold=thr,
        op=">",
        unit="nats",
        details=f"MCSPU({SIGMA_HI}) − MCSPU({SIGMA_LO}) = {delta:.4f}  (threshold: >{thr})",
    ))

    # 3. Statistical significance — one-sided Mann-Whitney U
    #    H₁: MCSPU scores at σ_hi are stochastically greater than at σ_lo
    u_stat, p_val = stats.mannwhitneyu(
        scores_hi, scores_lo, alternative="greater")
    thr = THRESHOLDS["max_mannwhitney_p"]
    results.append(TestResult(
        name="statistical_significance",
        passed=(p_val < thr),
        metric=float(p_val),
        threshold=thr,
        op="<",
        unit="p-value",
        details=f"Mann-Whitney U (one-sided) p={p_val:.2e}  U={u_stat:.0f}",
    ))

    # 4. Effect size — Cohen's d between σ_lo and σ_hi distributions
    d = _cohens_d(scores_hi, scores_lo)
    thr = THRESHOLDS["min_cohens_d"]
    results.append(TestResult(
        name="effect_size",
        passed=(d > thr),
        metric=d,
        threshold=thr,
        op=">",
        unit="σ units",
        details=f"Cohen's d = {d:.2f}  (threshold: >{thr}; d=0.2 small, 0.5 medium, 0.8 large)",
    ))

    return results


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _colored(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


def _pass_str(passed: bool) -> str:
    if passed:
        return _colored("PASS", _GREEN)
    return _colored("FAIL", _RED)


def print_report(reports: List[DatasetReport]) -> int:
    """Print the full test report.  Returns 1 if any unexpected outcome."""

    divider = "═" * 70
    print(f"\n{_BOLD}{divider}{_RESET}")
    print(f"{_BOLD}  MCSPU UNCERTAINTY TEST SUITE{_RESET}")
    print(f"{_BOLD}{divider}{_RESET}\n")

    # ── Sanity tests table ──────────────────────────────────────────────────
    sanity_names = [t.name for t in reports[0].sanity] if reports else []
    col_w = max(len(n) for n in sanity_names) + 2 if sanity_names else 30
    header_cols = "  ".join(f"{r.label:>10}" for r in reports)
    print(f"{_BOLD}SANITY TESTS  (mathematical invariants — all models must pass){_RESET}")
    print(f"  {'Test':{col_w}s}  {header_cols}")
    print("  " + "─" * (col_w + 2 + 12 * len(reports)))

    for i, name in enumerate(sanity_names):
        row = f"  {name:{col_w}s}  "
        row += "  ".join(
            f"{_pass_str(r.sanity[i].passed):>10}" for r in reports)
        print(row)

    # ── Sensitivity tests table ─────────────────────────────────────────────
    sensitivity_names = [
        t.name for t in reports[0].sensitivity] if reports else []
    col_w2 = max(len(n) for n in sensitivity_names) + \
        2 if sensitivity_names else 30
    print(f"\n{_BOLD}SIGNAL SENSITIVITY TESTS{_RESET}")
    print(f"  Expected to PASS: TSQA, HAR, Sleep EDF  |  Expected to FAIL: ECG-QA")
    print(f"  {'Test':{col_w2}s}  {header_cols}")
    print("  " + "─" * (col_w2 + 2 + 12 * len(reports)))

    for i, name in enumerate(sensitivity_names):
        row = f"  {name:{col_w2}s}  "
        row += "  ".join(
            f"{_pass_str(r.sensitivity[i].passed):>10}" for r in reports)
        print(row)

    # ── Threshold reference ─────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'Test':<30}  {'Threshold':>12}  {'Direction'}")
    print(f"{'─'*70}")
    if reports:
        for t in reports[0].sensitivity:
            print(f"  {t.name:<28}  {t.threshold:>12.4f}  {t.op}  ({t.unit})")

    # ── Per-model metric table ──────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'Metric values at σ={lo} vs σ={hi}':}".format(
        lo=SIGMA_LO, hi=SIGMA_HI))
    print(f"{'─'*70}")
    header = f"  {'Metric':<30}  " + \
        "  ".join(f"{r.label:>10}" for r in reports)
    print(header)
    for i, name in enumerate(sensitivity_names):
        vals = "  ".join(f"{r.sensitivity[i].metric:>10.4f}" for r in reports)
        print(f"  {name:<30}  {vals}")

    # ── Overall verdicts ────────────────────────────────────────────────────
    print(f"\n{_BOLD}OVERALL VERDICTS{_RESET}")
    print("─" * 70)
    unexpected = 0
    for r in reports:
        if r.expected_uses_signal:
            expected_str = _colored("USES SIGNAL (expected)",    _GREEN)
        else:
            expected_str = _colored("IGNORES SIGNAL (expected)", _YELLOW)

        if r.outcome_as_expected():
            verdict = _colored("✓  AS EXPECTED", _GREEN)
        else:
            verdict = _colored("✗  UNEXPECTED RESULT", _RED)
            unexpected += 1

        uses = r.sensitivity_passed()
        signal_str = (
            _colored("USES SIGNAL",    _GREEN) if uses else
            _colored("IGNORES SIGNAL", _RED)
        )
        print(f"  {r.label:<12}  {signal_str:<30}  {verdict}")

    print()
    if unexpected == 0:
        print(_colored("  All outcomes match expectations.  Exit 0.", _GREEN))
    else:
        print(
            _colored(f"  {unexpected} unexpected outcome(s).  Exit 1.", _RED))
    print()

    return 1 if unexpected > 0 else 0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PALETTE = {
    "TSQA":      "#4C72B0",
    "HAR":       "#DD8452",
    "Sleep EDF": "#55A868",
    "ECG-QA":   "#C44E52",
}

STYLE = {
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
}


def _plot_test_results(reports: List[DatasetReport], out_dir: Path):
    """
    Heatmap: rows = tests, columns = datasets.
    Sanity tests and signal sensitivity tests in two row groups.
    Green = PASS, red = FAIL, with cell annotations.
    """
    all_tests = [("sanity", t) for t in reports[0].sanity] + \
        [("sensitivity", t) for t in reports[0].sensitivity]
    test_labels = [t.name.replace("_", "\n") for _, t in all_tests]
    ds_labels = [r.label for r in reports]
    n_tests = len(all_tests)
    n_ds = len(ds_labels)

    matrix = np.zeros((n_tests, n_ds))
    for j, rep in enumerate(reports):
        for i, (cat, _) in enumerate(all_tests):
            tests = rep.sanity if cat == "sanity" else rep.sensitivity
            matrix[i, j] = 1.0 if tests[i % len(tests)].passed else 0.0

    fig, ax = plt.subplots(figsize=(max(6, 2 * n_ds), max(5, 0.7 * n_tests)))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_ds))
    ax.set_xticklabels(ds_labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_tests))
    ax.set_yticklabels(test_labels, fontsize=9)

    # Annotate cells
    for i in range(n_tests):
        for j in range(n_ds):
            passed = matrix[i, j] > 0.5
            ax.text(j, i, "PASS" if passed else "FAIL",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if passed else "white")

    # Separator line between sanity and sensitivity blocks
    n_sanity = sum(1 for cat, _ in all_tests if cat == "sanity")
    ax.axhline(n_sanity - 0.5, color="white", linewidth=2.5)

    # Group labels in margin
    ax.text(-0.7, (n_sanity - 1) / 2, "SANITY", va="center", ha="right",
            fontsize=9, rotation=90, color="gray", transform=ax.transData)
    ax.text(-0.7, n_sanity + (n_tests - n_sanity - 1) / 2, "SIGNAL\nSENSITIVITY",
            va="center", ha="right", fontsize=9, rotation=90, color="gray",
            transform=ax.transData)

    ax.set_title("MCSPU Test Results", fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    _save(fig, out_dir / "test_results")


def _plot_sensitivity_metrics(reports: List[DatasetReport], out_dir: Path):
    """
    4 subplots — one per signal sensitivity test.
    Each shows metric values as horizontal bars with a threshold line.
    """
    n = len(reports[0].sensitivity)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        axes = [axes]

    for col, test_idx in enumerate(range(n)):
        ax = axes[col]
        test_name = reports[0].sensitivity[test_idx].name
        thr = reports[0].sensitivity[test_idx].threshold
        op = reports[0].sensitivity[test_idx].op
        unit = reports[0].sensitivity[test_idx].unit

        labels = [r.label for r in reports]
        metrics = [r.sensitivity[test_idx].metric for r in reports]
        passed = [r.sensitivity[test_idx].passed for r in reports]
        colours = [PALETTE[l] for l in labels]
        alphas = [0.9 if p else 0.5 for p in passed]

        bars = ax.bar(range(len(labels)), metrics, color=colours, alpha=0.85)
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)

        # threshold line
        ax.axhline(thr, color="black", linewidth=1.5,
                   linestyle="--", label=f"threshold ({thr})")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_title(test_name.replace("_", " "), fontsize=10)
        if col == 0:
            ax.set_ylabel(f"Metric value ({unit})" if unit else "Metric value")
        ax.legend(fontsize=8, frameon=False)

        # PASS/FAIL badges
        for i, (metric, p) in enumerate(zip(metrics, passed)):
            badge = "✓" if p else "✗"
            color = "#2ca02c" if p else "#d62728"
            ax.text(i, metric + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    badge, ha="center", va="bottom", fontsize=13, color=color)

    fig.suptitle(f"Signal Sensitivity Metrics  (σ={SIGMA_LO} vs σ={SIGMA_HI})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "test_metrics")


def _plot_sigma_sweep(reports: List[DatasetReport],
                      sigma_data_all: dict,
                      out_dir: Path):
    """
    MCSPU vs σ line chart annotated with USES/IGNORES SIGNAL verdict.
    """
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(6, 4))

    for rep in reports:
        sdata = sigma_data_all.get(rep.stage_key, {})
        colour = PALETTE[rep.label]
        xs, ys, cis = [], [], []
        for sigma in SIGMAS:
            if sigma not in sdata:
                continue
            scores = np.array([r["mcspu_score"] for r in sdata[sigma]])
            xs.append(sigma)
            ys.append(scores.mean())
            cis.append(1.96 * scores.std() / np.sqrt(len(scores)))

        xs, ys, cis = np.array(xs), np.array(ys), np.array(cis)
        verdict = "USES SIGNAL" if rep.sensitivity_passed() else "IGNORES SIGNAL"
        linestyle = "-" if rep.sensitivity_passed() else "--"
        ax.plot(xs, ys, marker="o", label=f"{verdict}",
                color=colour, linewidth=2, linestyle=linestyle)
        ax.fill_between(xs, ys - cis, ys + cis, alpha=0.12, color=colour)

    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Mean MCSPU score (± 95% CI)")
    ax.set_title("Signal sensitivity: MCSPU vs noise magnitude")
    ax.set_xticks(SIGMAS)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    _save(fig, out_dir / "test_sigma_sweep")


def _save(fig, stem: Path):
    fig.savefig(str(stem) + ".pdf", dpi=150, bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MCSPU uncertainty test suite")
    parser.add_argument("--results_dir", default="mcspu_results", type=Path)
    parser.add_argument("--out_dir",     default="plots",         type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)

    # ── Load ────────────────────────────────────────────────────────────────
    sigma_data_all = {}
    reports: List[DatasetReport] = []

    print("\nLoading results ...")
    for stage_key, meta in DATASETS.items():
        sdata = load_stage(args.results_dir, stage_key)
        if sdata is None:
            print(f"  [warn] {stage_key} not found — skipping")
            continue
        sigma_data_all[stage_key] = sdata
        n_samples = sum(len(v) for v in sdata.values())
        print(
            f"  {meta['label']:<12}  {len(sdata)} sigma files  ({n_samples} total records)")

        report = DatasetReport(
            stage_key=stage_key,
            label=meta["label"],
            expected_uses_signal=meta["expected_uses_signal"],
            sanity=_run_sanity(sdata),
            sensitivity=_run_sensitivity(sdata),
        )
        reports.append(report)

    if not reports:
        print("No data found. Exiting.")
        sys.exit(1)

    # ── Terminal report ──────────────────────────────────────────────────────
    exit_code = print_report(reports)

    # ── Plots ────────────────────────────────────────────────────────────────
    print("Generating plots ...")
    _plot_test_results(reports, args.out_dir)
    _plot_sensitivity_metrics(reports, args.out_dir)
    _plot_sigma_sweep(reports, sigma_data_all, args.out_dir)
    print(f"Plots saved to {args.out_dir}/\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
