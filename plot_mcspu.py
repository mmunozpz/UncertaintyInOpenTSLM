#!/usr/bin/env python3
"""
plot_mcspu.py  —  Visualise MCSPU results for all stages / sigma sweeps.

Produces 8 figures saved to plots/:
  1. mcspu_vs_sigma.png              — Mean ± 95% CI MCSPU score vs noise σ per dataset
  2. distributions.png               — Violin plots of per-sample MCSPU at every σ
  3. heatmap.png                     — Mean MCSPU colour-coded (datasets × sigmas)
  4. mcspu_vs_conf_sigma_0.1.png     — MCSPU vs (1 − max clean prob) scatter, σ=0.1
  5. mcspu_vs_conf_sigma_0.5.png     — MCSPU vs (1 − max clean prob) scatter, σ=0.5
  6. mcspu_vs_conf_sigma_1.0.png     — MCSPU vs (1 − max clean prob) scatter, σ=1.0
  7. mcspu_vs_conf_sigma_2.0.png     — MCSPU vs (1 − max clean prob) scatter, σ=2.0
  8. table.png                       — Pearson r (MCSPU vs 1−max_prob) table: datasets × sigmas

Usage:
    python plot_mcspu.py
    python plot_mcspu.py --results_dir mcspu_results --out_dir plots
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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
    path = out_dir / "mcspu_vs_sigma.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
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
    path = out_dir / "distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
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
    path = out_dir / "heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 4: MCSPU vs model confidence (1 − max clean prob)
# ---------------------------------------------------------------------------

def plot_mcspu_vs_confidence(data: dict, out_dir: Path):
    for sigma in SIGMAS:
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

            corr = np.corrcoef(mcspu, uncertainty)[0, 1]
            ax.set_title(f"{label}\n(r = {corr:.2f})", fontsize=10)
            ax.set_xlabel("1 − max clean prob")
            if col == 0:
                ax.set_ylabel("MCSPU score")

            z = np.polyfit(uncertainty, mcspu, 1)
            xline = np.linspace(uncertainty.min(), uncertainty.max(), 100)
            ax.plot(xline, np.polyval(z, xline), color=colour, linewidth=1.5, linestyle="--")

        fig.suptitle(f"MCSPU vs model confidence (σ = {sigma})", fontsize=12)
        fig.tight_layout()
        path = out_dir / f"mcspu_vs_conf_sigma_{sigma}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 5: Pearson r table (datasets × sigmas)
# ---------------------------------------------------------------------------

def plot_r_table(data: dict, out_dir: Path):
    dataset_labels = list(STAGES.values())
    sigma_labels   = [f"σ={s}" for s in SIGMAS]

    # Build matrix of r values
    r_matrix = np.full((len(STAGES), len(SIGMAS)), np.nan)
    for i, (stage_key, label) in enumerate(STAGES.items()):
        if stage_key not in data:
            continue
        for j, sigma in enumerate(SIGMAS):
            if sigma not in data[stage_key]:
                continue
            records = data[stage_key][sigma]
            mcspu       = np.array([r["mcspu_score"] for r in records])
            uncertainty = np.array([1.0 - max(r["clean_probs"]) for r in records])
            r_matrix[i, j] = np.corrcoef(mcspu, uncertainty)[0, 1]

    fig, ax = plt.subplots(figsize=(5, 2.4))
    ax.axis("off")

    cell_text  = [[f"{r_matrix[i, j]:.2f}" if not np.isnan(r_matrix[i, j]) else "—"
                   for j in range(len(SIGMAS))]
                  for i in range(len(dataset_labels))]

    # Colour each cell by sign: red for negative, blue for positive
    cell_colours = []
    for i in range(len(dataset_labels)):
        row_colours = []
        for j in range(len(SIGMAS)):
            val = r_matrix[i, j]
            if np.isnan(val):
                row_colours.append("#f0f0f0")
            elif val < 0:
                intensity = min(abs(val), 1.0)
                r_ch = int(255 - (255 - 220) * intensity)
                g_ch = int(255 - (255 - 100) * intensity)
                b_ch = int(255 - (255 - 100) * intensity)
                row_colours.append(f"#{r_ch:02x}{g_ch:02x}{b_ch:02x}")
            else:
                intensity = min(abs(val), 1.0)
                r_ch = int(255 - (255 - 100) * intensity)
                g_ch = int(255 - (255 - 149) * intensity)
                b_ch = int(255 - (255 - 237) * intensity)
                row_colours.append(f"#{r_ch:02x}{g_ch:02x}{b_ch:02x}")
        cell_colours.append(row_colours)

    tbl = ax.table(
        cellText=cell_text,
        cellColours=cell_colours,
        rowLabels=dataset_labels,
        colLabels=sigma_labels,
        rowColours=[PALETTE[l] for l in dataset_labels],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)

    # Make row label text white for readability against the dataset colour
    for i, label in enumerate(dataset_labels):
        tbl[(i + 1, -1)].get_text().set_color("white")
        tbl[(i + 1, -1)].get_text().set_fontweight("bold")

    ax.set_title("Pearson r  (MCSPU vs 1 − max clean prob)", fontsize=11, pad=10)
    fig.tight_layout()
    path = out_dir / "table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
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
    plot_mcspu_vs_confidence(data, args.out_dir)
    plot_r_table(data, args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
