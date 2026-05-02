#!/usr/bin/env python3
"""
plot_mcspu.py  —  Visualise MCSPU results for all stages / sigma sweeps.

Produces 5 figures saved to plots/:
  1. mcspu_vs_sigma.pdf   — Mean ± 95% CI MCSPU score vs noise σ per dataset
  2. distributions.pdf    — Violin plots of per-sample MCSPU at every σ
  3. heatmap.pdf          — Mean MCSPU colour-coded (datasets × sigmas)
  4. accuracy.pdf         — Clean-signal accuracy per dataset
  5. mcspu_vs_conf.pdf    — MCSPU vs (1 − max clean prob) scatter per dataset

Usage:
    python plot_mcspu.py
    python plot_mcspu.py --results_dir mcspu_results --out_dir plots
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STAGES = {
    "stage1_tsqa_sp":   "TSQA",
    "stage3_har_sp":    "HAR",
    "stage4_sleep_sp":  "Sleep EDF",
    "stage5_ecg_sp":    "ECG-QA",
}
SIGMAS = [0.1, 0.5, 1.0, 2.0]

# Palette: one colour per dataset, consistent across all plots
PALETTE = {
    "TSQA":      "#4C72B0",
    "HAR":       "#DD8452",
    "Sleep EDF": "#55A868",
    "ECG-QA":   "#C44E52",
}

STYLE = {
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_gt_label(ground_truth: str, dataset: str) -> str:
    """Extract the short answer label from a (possibly CoT) ground_truth string."""
    gt = ground_truth.strip()

    if dataset == "tsqa":
        # format: "(b)<|end_of_text|>"  →  "B"
        m = re.search(r"\(([a-dA-D])\)", gt)
        return m.group(1).upper() if m else gt

    # HAR / Sleep / ECG: "… Answer: <label>[.]<|end_of_text|>"
    m = re.search(r"Answer:\s*(.+?)\.?\s*(?:<\|end_of_text\|>)?$", gt)
    if m:
        return m.group(1).strip()

    # fallback: strip EOS token
    return gt.replace("<|end_of_text|>", "").strip().rstrip(".")


def load_results(results_dir: Path) -> dict:
    """
    Returns:
        data[stage_key][sigma] = list of record dicts
    """
    data = {}
    for stage_key, label in STAGES.items():
        stage_dir = results_dir / stage_key
        if not stage_dir.exists():
            print(f"  [warn] {stage_dir} not found, skipping")
            continue
        data[stage_key] = {}
        for sigma in SIGMAS:
            path = stage_dir / f"sigma_{sigma}.jsonl"
            if not path.exists():
                print(f"  [warn] {path} not found, skipping")
                continue
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            data[stage_key][sigma] = records
    return data


# ---------------------------------------------------------------------------
# Figure 1: MCSPU mean ± 95% CI vs sigma
# ---------------------------------------------------------------------------

def plot_mcspu_vs_sigma(data: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))

    for stage_key, label in STAGES.items():
        if stage_key not in data:
            continue
        colour = PALETTE[label]
        means, cis, xs = [], [], []
        for sigma in SIGMAS:
            if sigma not in data[stage_key]:
                continue
            scores = np.array([r["mcspu_score"] for r in data[stage_key][sigma]])
            n = len(scores)
            means.append(scores.mean())
            cis.append(1.96 * scores.std() / np.sqrt(n))  # 95% CI
            xs.append(sigma)
        xs, means, cis = np.array(xs), np.array(means), np.array(cis)
        ax.plot(xs, means, marker="o", label=label, color=colour, linewidth=2)
        ax.fill_between(xs, means - cis, means + cis, alpha=0.15, color=colour)

    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Mean MCSPU score")
    ax.set_title("Signal sensitivity vs noise magnitude")
    ax.set_xticks(SIGMAS)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "mcspu_vs_sigma.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: Violin distributions of MCSPU at each sigma
# ---------------------------------------------------------------------------

def plot_distributions(data: dict, out_dir: Path):
    n_sigmas = len(SIGMAS)
    n_datasets = len(STAGES)
    fig, axes = plt.subplots(1, n_sigmas, figsize=(4 * n_sigmas, 4.5), sharey=False)

    for col, sigma in enumerate(SIGMAS):
        ax = axes[col]
        labels_order = list(STAGES.values())
        all_scores = []
        positions = []
        colours = []
        tick_labels = []

        for i, (stage_key, label) in enumerate(STAGES.items()):
            if stage_key not in data or sigma not in data[stage_key]:
                continue
            scores = [r["mcspu_score"] for r in data[stage_key][sigma]]
            all_scores.append(scores)
            positions.append(i + 1)
            colours.append(PALETTE[label])
            tick_labels.append(label)

        vp = ax.violinplot(all_scores, positions=positions, showmedians=True, showextrema=False)
        for body, col in zip(vp["bodies"], colours):
            body.set_facecolor(col)
            body.set_alpha(0.7)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"σ = {sigma}", fontsize=11)
        if col == 0:
            ax.set_ylabel("MCSPU score")

    fig.suptitle("Per-sample MCSPU distributions", fontsize=12, y=1.01)
    fig.tight_layout()
    path = out_dir / "distributions.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: Heatmap — mean MCSPU (datasets × sigmas)
# ---------------------------------------------------------------------------

def plot_heatmap(data: dict, out_dir: Path):
    labels = list(STAGES.values())
    matrix = np.full((len(STAGES), len(SIGMAS)), np.nan)

    for i, stage_key in enumerate(STAGES):
        if stage_key not in data:
            continue
        for j, sigma in enumerate(SIGMAS):
            if sigma not in data[stage_key]:
                continue
            scores = [r["mcspu_score"] for r in data[stage_key][sigma]]
            matrix[i, j] = np.mean(scores)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean MCSPU")

    ax.set_xticks(range(len(SIGMAS)))
    ax.set_xticklabels([f"σ={s}" for s in SIGMAS])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Mean MCSPU score (dataset × noise σ)")

    for i in range(len(STAGES)):
        for j in range(len(SIGMAS)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_col = "white" if val > matrix[~np.isnan(matrix)].mean() else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8.5, color=text_col, fontweight="bold")

    fig.tight_layout()
    path = out_dir / "heatmap.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 4: Clean-signal accuracy per dataset
# ---------------------------------------------------------------------------

def plot_accuracy(data: dict, out_dir: Path):
    # Use sigma=1.0 as the representative run; accuracy is independent of sigma
    sigma = 1.0
    labels, accs, errs = [], [], []

    for stage_key, label in STAGES.items():
        if stage_key not in data or sigma not in data[stage_key]:
            continue
        records = data[stage_key][sigma]
        dataset_name = records[0]["dataset"]
        correct = 0
        for r in records:
            pred = r["clean_pred"].strip().lower()
            gt = _extract_gt_label(r["ground_truth"], dataset_name).strip().lower()
            if pred == gt:
                correct += 1
        n = len(records)
        p = correct / n
        # Wilson confidence interval approximation
        z = 1.96
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        labels.append(label)
        accs.append(p)
        errs.append(margin)

    x = np.arange(len(labels))
    colours = [PALETTE[l] for l in labels]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(x, accs, yerr=errs, color=colours, alpha=0.85,
                  capsize=4, error_kw={"linewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (clean signal)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Clean-signal prediction accuracy")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = out_dir / "accuracy.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 5: MCSPU vs model confidence (1 − max clean prob)
# ---------------------------------------------------------------------------

def plot_mcspu_vs_confidence(data: dict, out_dir: Path):
    sigma = 1.0
    fig, axes = plt.subplots(1, len(STAGES), figsize=(4 * len(STAGES), 3.8), sharey=False)

    for col, (stage_key, label) in enumerate(STAGES.items()):
        ax = axes[col]
        colour = PALETTE[label]

        if stage_key not in data or sigma not in data[stage_key]:
            ax.set_visible(False)
            continue

        records = data[stage_key][sigma]
        mcspu  = np.array([r["mcspu_score"] for r in records])
        conf   = np.array([max(r["clean_probs"]) for r in records])
        uncertainty = 1.0 - conf

        ax.scatter(uncertainty, mcspu, alpha=0.4, s=18, color=colour, edgecolors="none")

        # Pearson correlation
        corr = np.corrcoef(mcspu, uncertainty)[0, 1]
        ax.set_title(f"{label}\n(r = {corr:.2f})", fontsize=10)
        ax.set_xlabel("1 − max clean prob")
        if col == 0:
            ax.set_ylabel("MCSPU score")

        # Trend line
        z = np.polyfit(uncertainty, mcspu, 1)
        xline = np.linspace(uncertainty.min(), uncertainty.max(), 100)
        ax.plot(xline, np.polyval(z, xline), color=colour, linewidth=1.5, linestyle="--")

    fig.suptitle("MCSPU vs model confidence (σ = 1.0)", fontsize=12)
    fig.tight_layout()
    path = out_dir / "mcspu_vs_conf.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="mcspu_results", type=Path)
    parser.add_argument("--out_dir",     default="plots",         type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(STYLE)

    print(f"Loading results from {args.results_dir} ...")
    data = load_results(args.results_dir)
    print(f"Loaded {sum(len(v) for v in data.values())} sigma×stage combinations\n")

    print("Generating plots ...")
    plot_mcspu_vs_sigma(data, args.out_dir)
    plot_distributions(data, args.out_dir)
    plot_heatmap(data, args.out_dir)
    plot_accuracy(data, args.out_dir)
    plot_mcspu_vs_confidence(data, args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
