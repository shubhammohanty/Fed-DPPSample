#!/usr/bin/env python3
"""
Stage 1 Hyperparameter Grid Search (Random Sampling / FedAvg).

Searches over:
  - learning-rate: [0.01, 0.005, 0.001, 0.0005]
  - local-epochs:  [1, 2, 3]
  - batch-size:    [16, 32]
  - fraction-fit:  [0.05, 0.1, 0.2, 0.3]

Outputs are organized into: results/grid_search_stage1/
Each run is saved with a descriptive filename, e.g.:
  run_lr0.001_ep2_bs32_ff0.1.json
  run_lr0.001_ep2_bs32_ff0.1.png

Resumption:
  If a run already has an output JSON in the destination directory, it is skipped.
"""

import os
import sys
import time
import shutil
import subprocess
import itertools
from pathlib import Path

# ==============================================================================
# CONFIGURATION & PATHS (Adjust if needed on your server)
# ==============================================================================
# Base results folder where server_app.py writes run outputs:
_SERVER_RES = Path("/driveB/FL/group9/flproject/Fed-DPPSample/results")
RESULTS_BASE_DIR = _SERVER_RES if _SERVER_RES.parent.exists() else (Path(__file__).resolve().parent / "results")

# Dedicated subfolder for Stage 1 Grid Search:
DEST_DIR = RESULTS_BASE_DIR / "grid_search_stage3"

# Hyperparameter Grids to search:
PARAM_GRID = {
    "learning-rate": [0.0005, 0.0002, 0.0001],
    "local-epochs":  [1],
    "batch-size":    [32],
    "fraction-fit":  [0.02, 0.04],
    "local-steps":   [5, 10],
    "hidden-dim":    [8, 16],
}

# Fixed parameters for Stage 1:
FIXED_CONFIG = {
    "sampling-mode": "random",
    "num-server-rounds": 100,      # Number of rounds per trial
    "target-accuracy": 99.0,
    "save-model": "false",         # Keep false during search to avoid disk bloat
}

# ==============================================================================
# RUNNER LOGIC
# ==============================================================================

def get_run_filename(lr, ep, bs, ff, ls, hd=None):
    """Format a clean, descriptive file prefix."""
    if hd is not None:
        return f"run_lr{lr}_ep{ep}_bs{bs}_ff{ff}_ls{ls}_hd{hd}"
    return f"run_lr{lr}_ep{ep}_bs{bs}_ff{ff}_ls{ls}"


def find_newest_files(directory, prefix="run_", extensions=(".json", ".png")):
    """Find the most recently modified run files in directory."""
    newest = {}
    for ext in extensions:
        files = list(directory.glob(f"{prefix}*{ext}"))
        if files:
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            newest[ext] = files[0]
    return newest


def run_single_combination(lr, ep, bs, ff, ls, trial_idx, total_trials, hd=None):
    if hd is None:
        hd = FIXED_CONFIG.get("hidden-dim", 16)
    run_name = get_run_filename(lr, ep, bs, ff, ls, hd if "hidden-dim" in PARAM_GRID else None)
    dest_json = DEST_DIR / f"{run_name}.json"
    dest_png = DEST_DIR / f"{run_name}.png"

    # Skip if already completed (enables clean resumption if interrupted)
    if dest_json.exists():
        print(f"\n[{trial_idx}/{total_trials}] ⏩ SKIPPING (Already exists): {run_name}")
        return

    print("\n" + "=" * 80)
    print(f"[{trial_idx}/{total_trials}] 🚀 STARTING: {run_name}")
    print(f"       Params: lr={lr}, epochs={ep}, batch_size={bs}, fraction_fit={ff}, local_steps={ls}, hidden_dim={hd}")
    print("=" * 80)

    # Snapshot existing files in RESULTS_BASE_DIR before run
    before_files = set(RESULTS_BASE_DIR.glob("run_*"))

    # Format run-config overrides as a single valid string: '<k1>=<v1> <k2>=<v2> ...'
    config_overrides = [
        f"learning-rate={lr}",
        f"local-epochs={ep}",
        f"batch-size={bs}",
        f"fraction-fit={ff}",
        f'sampling-mode="{FIXED_CONFIG["sampling-mode"]}"',
        f"num-server-rounds={FIXED_CONFIG['num-server-rounds']}",
        f"hidden-dim={hd}",
        f"local-steps={ls}",
        f"target-accuracy={FIXED_CONFIG['target-accuracy']}",
        f"save-model={str(FIXED_CONFIG['save-model']).lower()}",
    ]
    run_config_str = " ".join(config_overrides)

    # Build the flwr run command (include --stream to wait synchronously until completion)
    cmd = [
        "flwr", "run", ".",
        "--stream",
        "--run-config",
        run_config_str,
    ]

    start_time = time.time()
    try:
        # Execute Flower run synchronously
        res = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Trial {run_name} failed with exit code {e.returncode}")
        return
    except KeyboardInterrupt:
        print("\n🛑 Grid search manually interrupted by user.")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"⏱️ Trial finished in {elapsed:.1f}s")

    # Find the newly created JSON and PNG files in RESULTS_BASE_DIR
    after_files = set(RESULTS_BASE_DIR.glob("run_*"))
    new_files = after_files - before_files

    # Fallback: if before/after set diff missed it, look for newest file
    new_json = next((f for f in new_files if f.suffix == ".json"), None)
    new_png = next((f for f in new_files if f.suffix == ".png"), None)

    if not new_json:
        newest = find_newest_files(RESULTS_BASE_DIR, prefix="run_")
        new_json = newest.get(".json")
        new_png = newest.get(".png")

    # Move and rename to destination folder
    if new_json and new_json.exists():
        shutil.move(str(new_json), str(dest_json))
        print(f"💾 Saved: {dest_json.name}")

    if new_png and new_png.exists():
        shutil.move(str(new_png), str(dest_png))
        print(f"📊 Saved: {dest_png.name}")


def main():
    RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Generate all combinations
    keys = list(PARAM_GRID.keys())
    combinations = list(itertools.product(*PARAM_GRID.values()))
    total = len(combinations)

    print("=" * 80)
    print("🎯 FEDERATED LEARNING HYPERPARAMETER GRID SEARCH - STAGE 1 (RANDOM SAMPLING)")
    print("=" * 80)
    print(f"Total Trials to Run: {total}")
    print(f"Search Grid:")
    for k, v in PARAM_GRID.items():
        print(f"  - {k}: {v}")
    print(f"Fixed Config: {FIXED_CONFIG}")
    print(f"Output Directory: {DEST_DIR}")
    print("=" * 80)

    for i, values in enumerate(combinations, 1):
        param_dict = dict(zip(keys, values))
        run_single_combination(
            lr=param_dict["learning-rate"],
            ep=param_dict.get("local-epochs", 1),
            bs=param_dict.get("batch-size", 32),
            ff=param_dict["fraction-fit"],
            ls=param_dict.get("local-steps", 10),
            hd=param_dict.get("hidden-dim", 16),
            trial_idx=i,
            total_trials=total,
        )

    print("\n" + "=" * 80)
    print(f"✅ STAGE 1 GRID SEARCH COMPLETE! All results in: {DEST_DIR}")
    print("Now run 'python export_grid_summary.py' to generate the comparison spreadsheet.")
    print("=" * 80)


if __name__ == "__main__":
    main()
