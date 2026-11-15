#!/usr/bin/env python3
"""
Publication-Quality Figure Generation Pipeline.
Generates:
1. Demographic Disparities in Text-Conflict Gaslighting vs Baseline Accuracy
2. Typographic Prompt Injection Attack Success Rate
3. Layer-Wise Causal Activation Patching Circuit Tracing
4. Cross-Model Modality Arbitration Frontier
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def plot_demographic_conflict(results_dir, output_dir):
    csv_path = os.path.join(output_dir, "demographic_disparities.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    groups = df["Demographic Group"].tolist()

    # Parse percentages
    base_acc = [float(x.replace("%", "")) for x in df["Baseline Acc (%)"]]
    text_adopt = [float(x.replace("%", "")) for x in df["Text Conflict Adopted (%)"]]

    x = np.arange(len(groups))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    r1 = ax.bar(x - w/2, base_acc, w, label="Clean Baseline Accuracy (A1)", color="#2b5c8f", edgecolor="black", linewidth=0.6)
    r2 = ax.bar(x + w/2, text_adopt, w, label="Text-Conflict Adoption Rate (A3)", color="#c93b2b", edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Rate (%)")
    ax.set_title("Visual Baseline Accuracy vs. Misleading Text Adoption by Demographic Group")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim([0, 105])
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    for r in r1:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)
    for r in r2:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    out_file = os.path.join(output_dir, "fig_demographic_conflict.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[+] Saved {out_file}")

def plot_causal_patching(results_dir, output_dir):
    patch_csvs = glob.glob(os.path.join(results_dir, "causal_patching_seed_*.csv"))
    if not patch_csvs:
        return

    df_patch = pd.concat([pd.read_csv(f) for f in patch_csvs], ignore_index=True)
    layer_stats = df_patch.groupby("layer")["causal_recovery"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(layer_stats["layer"], layer_stats["mean"], color="#1f77b4", marker="o", label="Mean Causal Recovery")
    ax.fill_between(
        layer_stats["layer"],
        layer_stats["mean"] - layer_stats["std"],
        layer_stats["mean"] + layer_stats["std"],
        color="#1f77b4",
        alpha=0.2,
        label="±1 Std Dev"
    )

    ax.set_xlabel("Transformer Decoder Layer Index ($l$)")
    ax.set_ylabel("Visual Truth Causal Recovery")
    ax.set_title("Layer-Wise Causal Activation Patching: Decision Circuit Localization")
    ax.set_ylim([-0.1, 1.1])
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    out_file = os.path.join(output_dir, "fig_causal_circuit_tracing.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[+] Saved {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate publication visualizations")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing scored results")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save figures")
    args = parser.parse_args()
    out = args.output_dir or args.results_dir
    os.makedirs(out, exist_ok=True)
    plot_demographic_conflict(args.results_dir, out)
    plot_causal_patching(args.results_dir, out)

if __name__ == "__main__":
    main()
