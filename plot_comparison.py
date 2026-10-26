#!/usr/bin/env python3
"""
Publication-Quality Plotting Script for DPP vs Random Federated Learning.
Usage:
    python plot_comparison.py
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_plots():
    results_dir = Path("results")
    if not results_dir.exists():
        print("Error: results/ directory not found.")
        return

    random_runs = sorted(list(results_dir.glob("run_random_*.json")) + list(results_dir.glob("run_normal_*.json")))
    dpp_runs = sorted(list(results_dir.glob("run_dpp_*.json")))

    if not random_runs or not dpp_runs:
        print("Error: Missing run JSON files in results/. Please run both sampling modes (random and dpp) first.")
        return

    latest_random = random_runs[-1]
    latest_dpp = dpp_runs[-1]
    print(f"Loading Random Run: {latest_random.name}")
    print(f"Loading DPP Run:    {latest_dpp.name}")

    with open(latest_random) as f:
        r_norm = json.load(f)
    with open(latest_dpp) as f:
        r_dpp = json.load(f)

    norm_hist = r_norm["history"][1:]  # skip round 0
    dpp_hist = r_dpp["history"][1:]

    rounds = [r["round"] for r in norm_hist]
    norm_acc = [r["test_accuracy"] for r in norm_hist]
    dpp_acc = [r["test_accuracy"] for r in dpp_hist]
    norm_loss = [r["test_loss"] for r in norm_hist]
    dpp_loss = [r["test_loss"] for r in dpp_hist]
    norm_cov = [r["cumulative_coverage_percent"] for r in norm_hist]
    dpp_cov = [r["cumulative_coverage_percent"] for r in dpp_hist]
    norm_disp = [r["mean_spatial_distance"] for r in norm_hist]
    dpp_disp = [r["mean_spatial_distance"] for r in dpp_hist]

    # ── Figure 1: 3-Panel Paper Overview ─────────────────────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Accuracy Convergence
    axs[0].plot(rounds, norm_acc, "s--", label="FedAvg (Random)", color="#d62728", linewidth=2.2, markersize=5)
    axs[0].plot(rounds, dpp_acc, "o-", label="DPP-FedAvg (Ours)", color="#1f77b4", linewidth=2.2, markersize=5)
    axs[0].set_title("(a) Global Test Accuracy vs. Rounds", fontsize=13, fontweight="bold")
    axs[0].set_xlabel("Communication Round", fontsize=11)
    axs[0].set_ylabel("Test Accuracy (%)", fontsize=11)
    axs[0].grid(True, linestyle="--", alpha=0.6)
    axs[0].legend(fontsize=11)

    # (b) Cumulative Network Coverage
    axs[1].plot(rounds, norm_cov, "s--", label="FedAvg (Random)", color="#d62728", linewidth=2.2, markersize=5)
    axs[1].plot(rounds, dpp_cov, "o-", label="DPP-FedAvg (Ours)", color="#2ca02c", linewidth=2.2, markersize=5)
    axs[1].axhline(100.0, color="gray", linestyle=":", label="100% Full Network")
    axs[1].set_title("(b) Cumulative Network Coverage (%)", fontsize=13, fontweight="bold")
    axs[1].set_xlabel("Communication Round", fontsize=11)
    axs[1].set_ylabel("Unique Sensors Sampled (%)", fontsize=11)
    axs[1].set_ylim([0, 105])
    axs[1].grid(True, linestyle="--", alpha=0.6)
    axs[1].legend(fontsize=11, loc="lower right")

    # (c) Spatial Dispersion
    axs[2].plot(rounds, norm_disp, "s--", label="FedAvg (Random)", color="#d62728", linewidth=2.2, markersize=5)
    axs[2].plot(rounds, dpp_disp, "o-", label="DPP-FedAvg (Ours)", color="#9467bd", linewidth=2.2, markersize=5)
    axs[2].set_title("(c) Spatial Dispersion of Sampled Clients", fontsize=13, fontweight="bold")
    axs[2].set_xlabel("Communication Round", fontsize=11)
    axs[2].set_ylabel("Mean Pairwise Shortest-Path Distance", fontsize=11)
    axs[2].grid(True, linestyle="--", alpha=0.6)
    axs[2].legend(fontsize=11)

    plt.tight_layout()
    overview_path = results_dir / "paper_comparison_3panel.png"
    fig.savefig(overview_path, dpi=300)
    plt.close(fig)

    # ── Figure 2: Detailed 4-Panel Loss & Client Fairness ────────────────────
    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Loss
    axs2[0, 0].plot(rounds, norm_loss, "s--", label="Random Test Loss", color="#d62728")
    axs2[0, 0].plot(rounds, dpp_loss, "o-", label="DPP Test Loss", color="#1f77b4")
    axs2[0, 0].set_title("Test MSE Loss vs Communication Round", fontweight="bold")
    axs2[0, 0].set_xlabel("Round")
    axs2[0, 0].set_ylabel("MSE Loss")
    axs2[0, 0].grid(True, linestyle="--", alpha=0.6)
    axs2[0, 0].legend()

    # Panel 2: Accuracy
    axs2[0, 1].plot(rounds, norm_acc, "s--", label="Random Accuracy", color="#d62728")
    axs2[0, 1].plot(rounds, dpp_acc, "o-", label="DPP Accuracy", color="#2ca02c")
    axs2[0, 1].set_title("Test Accuracy (%) vs Communication Round", fontweight="bold")
    axs2[0, 1].set_xlabel("Round")
    axs2[0, 1].set_ylabel("Accuracy (%)")
    axs2[0, 1].grid(True, linestyle="--", alpha=0.6)
    axs2[0, 1].legend()

    # Panel 3: Coverage
    axs2[1, 0].plot(rounds, norm_cov, "s--", label="Random Coverage", color="#d62728")
    axs2[1, 0].plot(rounds, dpp_cov, "o-", label="DPP Coverage", color="#9467bd")
    axs2[1, 0].axhline(100.0, color="gray", linestyle=":")
    axs2[1, 0].set_title("Cumulative Network Coverage (%)", fontweight="bold")
    axs2[1, 0].set_xlabel("Round")
    axs2[1, 0].set_ylabel("Sensors Visited (%)")
    axs2[1, 0].grid(True, linestyle="--", alpha=0.6)
    axs2[1, 0].legend(loc="lower right")

    # Panel 4: Client Selection Frequency Histogram (Fairness)
    norm_clients = [cid for r in norm_hist for cid in r["selected_clients"]]
    dpp_clients = [cid for r in dpp_hist for cid in r["selected_clients"]]
    norm_counts = np.bincount(norm_clients, minlength=228)
    dpp_counts = np.bincount(dpp_clients, minlength=228)

    axs2[1, 1].hist(norm_counts, bins=range(0, 15), alpha=0.5, label="Random (Unvisited: {})".format((norm_counts == 0).sum()), color="#d62728")
    axs2[1, 1].hist(dpp_counts, bins=range(0, 15), alpha=0.5, label="DPP (Unvisited: {})".format((dpp_counts == 0).sum()), color="#1f77b4")
    axs2[1, 1].set_title("Sensor Selection Frequency Distribution (Fairness)", fontweight="bold")
    axs2[1, 1].set_xlabel("Times Selected in 25 Rounds")
    axs2[1, 1].set_ylabel("Number of Sensors")
    axs2[1, 1].grid(True, linestyle="--", alpha=0.6)
    axs2[1, 1].legend()

    plt.tight_layout()
    detailed_path = results_dir / "paper_comparison_detailed.png"
    fig2.savefig(detailed_path, dpi=300)
    plt.close(fig2)

    print(f"\n✅ Publication figures successfully generated:")
    print(f"   1. {overview_path}")
    print(f"   2. {detailed_path}\n")

if __name__ == "__main__":
    generate_comparison_plots()
