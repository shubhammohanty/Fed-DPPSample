#!/usr/bin/env python3
"""
Publication-Quality Figure & Table Generator for DPP Paper.

Generates:
  1. fig1_convergence_comparison.png: Random vs DPP with speedup metrics
  2. fig2_parameter_sensitivity.png: Sensitivity to sigma (diversity) and alpha (starvation)
  3. fig3_fairness_and_dispersion.png: Cumulative Network Coverage (%) and Spatial Dispersion
  4. paper_benchmark_table.csv: Summary table ready for paper insertion
"""

import os
import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Styling for academic papers (NeurIPS / IEEE Transactions style)
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})

COLORS = {
    "random": "#d62728",     # Red
    "dpp_default": "#1f77b4",# Blue
    "sigma1.0": "#ff7f0e",   # Orange
    "sigma5.0": "#2ca02c",   # Green
    "alpha0.1": "#9467bd",   # Purple
    "alpha1.0": "#8c564b",   # Brown
    "pagerank": "#e377c2",   # Pink
}


def load_run(json_path: Path):
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    history = [r for r in data.get("history", []) if r.get("round", 0) > 0]
    return {"meta": meta, "history": history}


def get_trajectory(history):
    rounds = [r["round"] for r in history]
    acc = [r["test_accuracy"] for r in history]
    loss = [r["test_loss"] for r in history]
    cov = [r["cumulative_coverage_percent"] for r in history]
    disp = [r["mean_spatial_distance"] for r in history]
    return rounds, acc, loss, cov, disp


def round_to_acc(history, target):
    for r in history:
        if r.get("test_accuracy", 0.0) >= target:
            return r.get("round")
    return None


def plot_suite(results_dir: Path):
    output_dir = results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load runs
    runs = {
        "random": load_run(results_dir / "random_baseline.json"),
        "dpp_default": load_run(results_dir / "dpp_default.json"),
        "sigma1.0": load_run(results_dir / "dpp_sigma1.0.json"),
        "sigma5.0": load_run(results_dir / "dpp_sigma5.0.json"),
        "alpha0.1": load_run(results_dir / "dpp_alpha0.1.json"),
        "alpha1.0": load_run(results_dir / "dpp_alpha1.0.json"),
        "pagerank": load_run(results_dir / "dpp_pagerank.json"),
    }

    # Fallback to check results/grid_search_stage1 if random_baseline isn't in paper_experiments yet
    if runs["random"] is None:
        fallback_random = results_dir.parent / "grid_search_stage1" / "run_lr0.0005_ep1_bs32_ff0.04_ls10_hd16.json"
        if fallback_random.exists():
            print(f"Using fallback random run from: {fallback_random}")
            runs["random"] = load_run(fallback_random)

    available = {k: v for k, v in runs.items() if v is not None}
    print(f"Loaded {len(available)} / {len(runs)} experiment runs.")
    if not available:
        print("No run JSON files found! Run run_stage2_dpp_suite.py first.")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 1: Main Convergence Comparison (Random vs DPP Default)
    # ──────────────────────────────────────────────────────────────────────────
    if "random" in available and "dpp_default" in available:
        r_rnd, r_acc, r_lss, _, _ = get_trajectory(available["random"]["history"])
        d_rnd, d_acc, d_lss, _, _ = get_trajectory(available["dpp_default"]["history"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        # (a) Test Accuracy
        ax1.plot(r_rnd, r_acc, color=COLORS["random"], label="FedAvg (Random Sampling)", lw=2.2, linestyle="--")
        ax1.plot(d_rnd, d_acc, color=COLORS["dpp_default"], label="Proposed DPP (Ours)", lw=2.5)
        
        # Target threshold lines
        ax1.axhline(88.0, color="gray", linestyle=":", alpha=0.7, label="88% Target Accuracy")
        ax1.axhline(90.0, color="dimgray", linestyle="-.", alpha=0.7, label="90% Target Accuracy")

        r_to_88_rand = round_to_acc(available["random"]["history"], 88.0)
        r_to_88_dpp = round_to_acc(available["dpp_default"]["history"], 88.0)
        if r_to_88_rand and r_to_88_dpp:
            speedup = ((r_to_88_rand - r_to_88_dpp) / r_to_88_rand) * 100.0
            ax1.annotate(
                f"DPP reaches 88% in {r_to_88_dpp} rds\n(Random: {r_to_88_rand} rds, {speedup:.1f}% speedup)",
                xy=(r_to_88_dpp, 88.0),
                xytext=(r_to_88_dpp + 8, 83.0),
                arrowprops=dict(facecolor=COLORS["dpp_default"], shrink=0.05, width=1.5, headwidth=6),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["dpp_default"], alpha=0.9),
                fontsize=9.5,
            )

        ax1.set_title("(a) Global Test Accuracy (%) vs. Round")
        ax1.set_xlabel("Communication Round")
        ax1.set_ylabel("Test Accuracy (%)")
        ax1.set_ylim([20, 96])
        ax1.grid(True)
        ax1.legend(loc="lower right")

        # (b) Test Loss (MSE)
        ax2.plot(r_rnd, r_lss, color=COLORS["random"], label="FedAvg (Random)", lw=2.2, linestyle="--")
        ax2.plot(d_rnd, d_lss, color=COLORS["dpp_default"], label="Proposed DPP (Ours)", lw=2.5)
        ax2.set_title("(b) Centralized Test MSE Loss vs. Round")
        ax2.set_xlabel("Communication Round")
        ax2.set_ylabel("Mean Squared Error (MSE)")
        ax2.grid(True)
        ax2.legend(loc="upper right")

        plt.tight_layout()
        fig1_path = output_dir / "fig1_convergence_comparison.png"
        fig.savefig(fig1_path)
        plt.close(fig)
        print(f"✅ Generated: {fig1_path.name}")

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 2: Hyperparameter Sensitivity (Sigma & Alpha Studies)
    # ──────────────────────────────────────────────────────────────────────────
    fig, (ax_sig, ax_alp) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    # Panel (a): Sigma Sensitivity
    if "random" in available:
        r_rnd, r_acc, _, _, _ = get_trajectory(available["random"]["history"])
        ax_sig.plot(r_rnd, r_acc, color=COLORS["random"], label="Random (Baseline)", lw=1.8, linestyle=":")

    for k, label, col, ls in [
        ("sigma1.0", r"$\sigma = 1.0$ (Sharp Repulsion)", COLORS["sigma1.0"], "--"),
        ("dpp_default", r"$\sigma = 2.0$ (Optimal Default)", COLORS["dpp_default"], "-"),
        ("sigma5.0", r"$\sigma = 5.0$ (Broad Repulsion)", COLORS["sigma5.0"], "-."),
    ]:
        if k in available:
            rnd, acc, _, _, _ = get_trajectory(available[k]["history"])
            ax_sig.plot(rnd, acc, color=col, label=label, lw=2.2, linestyle=ls)

    ax_sig.set_title(r"(a) Sensitivity to Spatial Bandwidth $\sigma$ ($\alpha=0.5$)")
    ax_sig.set_xlabel("Communication Round")
    ax_sig.set_ylabel("Global Test Accuracy (%)")
    ax_sig.set_ylim([30, 96])
    ax_sig.grid(True)
    ax_sig.legend(loc="lower right")

    # Panel (b): Alpha Sensitivity
    if "random" in available:
        r_rnd, r_acc, _, _, _ = get_trajectory(available["random"]["history"])
        ax_alp.plot(r_rnd, r_acc, color=COLORS["random"], label="Random (Baseline)", lw=1.8, linestyle=":")

    for k, label, col, ls in [
        ("alpha0.1", r"$\alpha = 0.1$ (Low Starvation Penalty)", COLORS["alpha0.1"], "--"),
        ("dpp_default", r"$\alpha = 0.5$ (Optimal Default)", COLORS["dpp_default"], "-"),
        ("alpha1.0", r"$\alpha = 1.0$ (High Starvation Penalty)", COLORS["alpha1.0"], "-."),
    ]:
        if k in available:
            rnd, acc, _, _, _ = get_trajectory(available[k]["history"])
            ax_alp.plot(rnd, acc, color=col, label=label, lw=2.2, linestyle=ls)

    ax_alp.set_title(r"(b) Sensitivity to Starvation Penalty $\alpha$ ($\sigma=2.0$)")
    ax_alp.set_xlabel("Communication Round")
    ax_alp.set_ylabel("Global Test Accuracy (%)")
    ax_alp.set_ylim([30, 96])
    ax_alp.grid(True)
    ax_alp.legend(loc="lower right")

    plt.tight_layout()
    fig2_path = output_dir / "fig2_parameter_sensitivity.png"
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f"✅ Generated: {fig2_path.name}")

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 3: Fairness (Network Coverage) & Spatial Dispersion
    # ──────────────────────────────────────────────────────────────────────────
    fig, (ax_cov, ax_dsp) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    # Panel (a): Cumulative Network Coverage (%)
    for k, label, col, ls in [
        ("random", "FedAvg (Random)", COLORS["random"], ":"),
        ("dpp_default", "DPP (Default)", COLORS["dpp_default"], "-"),
        ("alpha0.1", r"DPP ($\alpha=0.1$)", COLORS["alpha0.1"], "--"),
        ("alpha1.0", r"DPP ($\alpha=1.0$)", COLORS["alpha1.0"], "-."),
    ]:
        if k in available:
            rnd, _, _, cov, _ = get_trajectory(available[k]["history"])
            ax_cov.plot(rnd, cov, color=col, label=label, lw=2.2, linestyle=ls)

    ax_cov.axhline(100.0, color="gray", linestyle="--", alpha=0.5, label="100% Complete Network Coverage")
    ax_cov.set_title("(a) Cumulative Network Coverage (% of 228 Sensors)")
    ax_cov.set_xlabel("Communication Round")
    ax_cov.set_ylabel("Sensors Sampled (%)")
    ax_cov.set_ylim([0, 105])
    ax_cov.grid(True)
    ax_cov.legend(loc="lower right")

    # Panel (b): Spatial Dispersion (Mean Shortest-Path Distance)
    for k, label, col, ls in [
        ("random", "FedAvg (Random)", COLORS["random"], ":"),
        ("dpp_default", r"DPP ($\sigma=2.0$)", COLORS["dpp_default"], "-"),
        ("sigma1.0", r"DPP ($\sigma=1.0$)", COLORS["sigma1.0"], "--"),
        ("sigma5.0", r"DPP ($\sigma=5.0$)", COLORS["sigma5.0"], "-."),
    ]:
        if k in available:
            rnd, _, _, _, dsp = get_trajectory(available[k]["history"])
            valid_idx = [i for i, d in enumerate(dsp) if d is not None]
            if valid_idx:
                ax_dsp.plot([rnd[i] for i in valid_idx], [dsp[i] for i in valid_idx], color=col, label=label, lw=2.0, linestyle=ls)

    ax_dsp.set_title("(b) Spatial Dispersion (Mean Pairwise Network Distance)")
    ax_dsp.set_xlabel("Communication Round")
    ax_dsp.set_ylabel("Graph Distance (hops / weighted)")
    ax_dsp.grid(True)
    ax_dsp.legend(loc="upper right")

    plt.tight_layout()
    fig3_path = output_dir / "fig3_fairness_and_dispersion.png"
    fig.savefig(fig3_path)
    plt.close(fig)
    print(f"✅ Generated: {fig3_path.name}")

    # ──────────────────────────────────────────────────────────────────────────
    # TABLE 1: Comprehensive Benchmark Table
    # ──────────────────────────────────────────────────────────────────────────
    table_rows = []
    for k, name_label in [
        ("random", "FedAvg (Uniform Random)"),
        ("dpp_default", "Proposed DPP (Default)"),
        ("sigma1.0", "DPP (sigma=1.0)"),
        ("sigma5.0", "DPP (sigma=5.0)"),
        ("alpha0.1", "DPP (alpha=0.1)"),
        ("alpha1.0", "DPP (alpha=1.0)"),
        ("pagerank", "DPP (PageRank Centrality)"),
    ]:
        if k not in available:
            continue
        data = available[k]
        meta = data["meta"]
        hist = data["history"]
        rnd, acc, lss, cov, dsp = get_trajectory(hist)
        
        valid_dsp = [d for d in dsp if d is not None]
        mean_dsp = np.mean(valid_dsp) if valid_dsp else None

        table_rows.append({
            "Method / Configuration": name_label,
            "Sampling": meta.get("sampling-mode", "random"),
            "sigma": meta.get("sigma", "-"),
            "alpha": meta.get("alpha", "-"),
            "Centrality": meta.get("centrality-type", "-"),
            "Rounds to 85%": round_to_acc(hist, 85.0),
            "Rounds to 88%": round_to_acc(hist, 88.0),
            "Rounds to 90%": round_to_acc(hist, 90.0),
            "Peak Acc (%)": round(max(acc), 2) if acc else None,
            "Final Loss": round(lss[-1], 4) if lss else None,
            "Coverage (%)": round(cov[-1], 1) if cov else None,
            "Mean Spatial Dist": round(mean_dsp, 2) if mean_dsp else None,
        })

    df_table = pd.DataFrame(table_rows)
    table_path = output_dir / "paper_benchmark_table.csv"
    df_table.to_csv(table_path, index=False)
    print(f"✅ Generated: {table_path.name}")

    print("\n" + "=" * 100)
    print("📊 PAPER BENCHMARK SUMMARY TABLE:")
    print("=" * 100)
    print(df_table.to_string(index=False))
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="/driveB/FL/group9/flproject/Fed-DPPSample/results/paper_experiments")
    args = parser.parse_args()

    rdir = Path(args.results_dir)
    if not rdir.exists():
        # Check local path fallback
        local_dir = Path(__file__).resolve().parent / "results" / "paper_experiments"
        if not local_dir.exists():
            local_dir = Path("/home/shubham/Desktop/FL/IntSampling/results/paper_experiments")
        if local_dir.exists():
            rdir = local_dir
        else:
            print(f"Directory {rdir} does not exist yet. Please run run_stage2_dpp_suite.py first.")
            return

    plot_suite(rdir)


if __name__ == "__main__":
    main()
