#!/usr/bin/env python3
"""
Stage 2 DPP Experiment Suite & Hyperparameter Study.

Executes a focused set of 7 runs for the paper in ~15-20 minutes:
  1. Baseline Random (FedAvg)
  2. Proposed DPP (Default: sigma=2.0, alpha=0.5, degree)
  3. sigma ablation: sigma=1.0 (narrow/local diversity)
  4. sigma ablation: sigma=5.0 (wide/global diversity)
  5. alpha ablation: alpha=0.1 (low starvation penalty)
  6. alpha ablation: alpha=1.0 (high starvation penalty)
  7. Centrality ablation: PageRank vs Degree (sigma=2.0, alpha=0.5, pagerank)

Fixed Training Hyperparameters (from Stage 1 Grid Search):
  - learning-rate: 0.0005
  - fraction-fit:  0.04  (k ≈ 9 clients per round)
  - local-steps:   10
  - hidden-dim:    16
  - batch-size:    32
  - local-epochs:  1
  - rounds:        100
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
_SERVER_RES = Path("/driveB/FL/group9/flproject/Fed-DPPSample/results")
RESULTS_BASE_DIR = (_SERVER_RES if _SERVER_RES.parent.exists() else (Path(__file__).resolve().parent / "results")).resolve()
DEST_DIR = RESULTS_BASE_DIR / "paper_experiments"

# Fixed Training Hyperparameters from Stage 1:
FIXED_PARAMS = {
    "learning-rate": 0.0005,
    "fraction-fit":  0.04,
    "local-steps":   10,
    "hidden-dim":    16,
    "batch-size":    32,
    "local-epochs":  1,
    "num-server-rounds": 100,
    "target-accuracy": 99.0,
    "save-model": "false",
}

# 7 Focused Runs for Paper:
EXPERIMENTS = [
    {
        "name": "random_baseline",
        "sampling-mode": "random",
        "sigma": 2.0,
        "alpha": 0.5,
        "centrality-type": "degree",
        "desc": "Baseline FedAvg (Uniform Random Sampling)"
    },
    {
        "name": "dpp_default",
        "sampling-mode": "dpp",
        "sigma": 2.0,
        "alpha": 0.5,
        "centrality-type": "degree",
        "desc": "Proposed DPP (sigma=2.0, alpha=0.5, Degree)"
    },
    {
        "name": "dpp_sigma1.0",
        "sampling-mode": "dpp",
        "sigma": 1.0,
        "alpha": 0.5,
        "centrality-type": "degree",
        "desc": "DPP sigma=1.0 (Sharp spatial repulsion)"
    },
    {
        "name": "dpp_sigma5.0",
        "sampling-mode": "dpp",
        "sigma": 5.0,
        "alpha": 0.5,
        "centrality-type": "degree",
        "desc": "DPP sigma=5.0 (Broad spatial repulsion)"
    },
    {
        "name": "dpp_alpha0.1",
        "sampling-mode": "dpp",
        "sigma": 2.0,
        "alpha": 0.1,
        "centrality-type": "degree",
        "desc": "DPP alpha=0.1 (Low starvation penalty)"
    },
    {
        "name": "dpp_alpha1.0",
        "sampling-mode": "dpp",
        "sigma": 2.0,
        "alpha": 1.0,
        "centrality-type": "degree",
        "desc": "DPP alpha=1.0 (High starvation penalty)"
    },
    {
        "name": "dpp_pagerank",
        "sampling-mode": "dpp",
        "sigma": 2.0,
        "alpha": 0.5,
        "centrality-type": "pagerank",
        "desc": "DPP with PageRank Centrality"
    },
]

# ==============================================================================
# RUNNER LOGIC
# ==============================================================================

def find_newest_files(directory, prefix="run_", extensions=(".json", ".png")):
    newest = {}
    for ext in extensions:
        files = list(directory.glob(f"{prefix}*{ext}"))
        if files:
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            newest[ext] = files[0]
    return newest


def run_experiment(exp_cfg, idx, total):
    name = exp_cfg["name"]
    dest_json = DEST_DIR / f"{name}.json"
    dest_png = DEST_DIR / f"{name}.png"

    if dest_json.exists():
        print(f"\n[{idx}/{total}] ⏩ SKIPPING (Already exists): {name}")
        return

    print("\n" + "=" * 80)
    print(f"[{idx}/{total}] 🚀 STARTING: {name}")
    print(f"       Description: {exp_cfg['desc']}")
    print(f"       Mode={exp_cfg['sampling-mode']}, sigma={exp_cfg['sigma']}, alpha={exp_cfg['alpha']}, centrality={exp_cfg['centrality-type']}")
    print("=" * 80)

    before_files = set(RESULTS_BASE_DIR.glob("run_*"))

    config_overrides = [
        f"learning-rate={FIXED_PARAMS['learning-rate']}",
        f"local-epochs={FIXED_PARAMS['local-epochs']}",
        f"batch-size={FIXED_PARAMS['batch-size']}",
        f"fraction-fit={FIXED_PARAMS['fraction-fit']}",
        f"num-server-rounds={FIXED_PARAMS['num-server-rounds']}",
        f"hidden-dim={FIXED_PARAMS['hidden-dim']}",
        f"local-steps={FIXED_PARAMS['local-steps']}",
        f"target-accuracy={FIXED_PARAMS['target-accuracy']}",
        f"save-model={FIXED_PARAMS['save-model']}",
        f'sampling-mode="{exp_cfg["sampling-mode"]}"',
        f"sigma={exp_cfg['sigma']}",
        f"alpha={exp_cfg['alpha']}",
        f'centrality-type="{exp_cfg["centrality-type"]}"',
    ]
    run_config_str = " ".join(config_overrides)

    cmd = [
        "flwr", "run", ".",
        "--stream",
        "--run-config",
        run_config_str,
    ]

    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment {name} failed with exit code {e.returncode}")
        return
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"⏱️ Finished in {elapsed:.1f}s")

    after_files = set(RESULTS_BASE_DIR.glob("run_*"))
    new_files = after_files - before_files

    new_json = next((f for f in new_files if f.suffix == ".json"), None)
    new_png = next((f for f in new_files if f.suffix == ".png"), None)

    if not new_json:
        newest = find_newest_files(RESULTS_BASE_DIR, prefix="run_")
        new_json = newest.get(".json")
        new_png = newest.get(".png")

    if new_json and new_json.exists():
        shutil.move(str(new_json), str(dest_json))
        print(f"💾 Saved JSON: {dest_json.name}")

    if new_png and new_png.exists():
        shutil.move(str(new_png), str(dest_png))
        print(f"📊 Saved Diagnostic Plot: {dest_png.name}")


def main():
    RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    total = len(EXPERIMENTS)
    print("=" * 80)
    print("🎯 STAGE 2: DPP HYPERPARAMETER STUDY & BENCHMARK SUITE")
    print(f"Total Runs: {total} (Est. Runtime: ~15-20 minutes total)")
    print(f"Output Directory: {DEST_DIR}")
    print("=" * 80)

    for i, exp in enumerate(EXPERIMENTS, 1):
        run_experiment(exp, i, total)

    print("\n" + "=" * 80)
    print(f"✅ ALL PAPER RUNS COMPLETED! Saved to: {DEST_DIR}")
    print("Now run: python plot_paper_suite.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
