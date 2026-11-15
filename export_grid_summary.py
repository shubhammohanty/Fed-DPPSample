#!/usr/bin/env python3
"""
Grid Search Results Aggregator & Spreadsheet Exporter.

Reads all run JSON logs from an experiment directory (e.g. results/grid_search_stage1),
extracts all hyperparameters and convergence metrics, sorts configs from best to worst,
and exports a clean spreadsheet (CSV) ready for Excel / Google Sheets / Paper Tables.

Usage:
    python export_grid_summary.py
    python export_grid_summary.py --input-dir results/grid_search_stage1
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd

def parse_run_json(json_file: Path) -> dict:
    with open(json_file, "r") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    history = data.get("history", [])

    # Exclude initial evaluation (Round 0) for convergence stats if history has multiple rounds
    rounds_data = [r for r in history if r.get("round", 0) > 0]
    if not rounds_data:
        rounds_data = history

    if not rounds_data:
        return None

    # Accuracy trajectory
    accuracies = [r.get("test_accuracy", 0.0) for r in rounds_data]
    losses = [r.get("test_loss", float("inf")) for r in rounds_data]
    maes = [r.get("test_mae", float("inf")) for r in rounds_data]

    best_acc = max(accuracies) if accuracies else 0.0
    best_acc_idx = accuracies.index(best_acc)
    best_acc_round = rounds_data[best_acc_idx].get("round", 0)

    final_acc = accuracies[-1] if accuracies else 0.0
    best_loss = min(losses) if losses else float("inf")
    final_loss = losses[-1] if losses else float("inf")
    best_mae = min(maes) if maes else float("inf")
    final_mae = maes[-1] if maes else float("inf")

    # Milestone rounds
    def get_round_to_target(target):
        for r in rounds_data:
            if r.get("test_accuracy", 0.0) >= target:
                return r.get("round")
        return None

    final_coverage = rounds_data[-1].get("cumulative_coverage_percent", 0.0)

    record = {
        "trial_name": json_file.stem,
        # Hyperparameters
        "learning_rate": meta.get("learning-rate"),
        "local_epochs": meta.get("local-epochs"),
        "batch_size": meta.get("batch-size"),
        "fraction_fit": meta.get("fraction-fit"),
        "hidden_dim": meta.get("hidden-dim"),
        "local_steps": meta.get("local-steps"),
        "sampling_mode": meta.get("sampling-mode"),
        "sigma": meta.get("sigma"),
        "alpha": meta.get("alpha"),
        "max_rounds": meta.get("num-server-rounds"),
        "completed_rounds": len(rounds_data),
        # Performance Metrics
        "best_accuracy_pct": round(best_acc, 2),
        "best_accuracy_round": best_acc_round,
        "final_accuracy_pct": round(final_acc, 2),
        "best_test_loss": round(best_loss, 4),
        "final_test_loss": round(final_loss, 4),
        "best_test_mae": round(best_mae, 4),
        "final_test_mae": round(final_mae, 4),
        "final_coverage_pct": round(final_coverage, 1),
        # Convergence Milestones
        "rounds_to_80pct": get_round_to_target(80.0),
        "rounds_to_85pct": get_round_to_target(85.0),
        "rounds_to_88pct": get_round_to_target(88.0),
        "rounds_to_90pct": get_round_to_target(90.0),
        "json_path": str(json_file.name),
    }
    return record


def main():
    parser = argparse.ArgumentParser(description="Export Grid Search results to Spreadsheet.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results/grid_search_stage1",
        help="Directory containing run_*.json files (default: results/grid_search_stage1)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path for exported CSV (default: <input-dir>/grid_search_summary.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.exists():
        # Fallback to root results/ if user hasn't created the subfolder yet
        if Path("results").exists():
            print(f"Directory {input_path} not found. Checking results/ directory...")
            input_path = Path("results")
        else:
            print(f"Error: Directory '{input_path}' not found.")
            return

    json_files = sorted(list(input_path.glob("run_*.json")))
    if not json_files:
        print(f"No run_*.json files found in '{input_path}'.")
        return

    print(f"Found {len(json_files)} run JSON files in: {input_path}")
    records = []
    for jf in json_files:
        try:
            r = parse_run_json(jf)
            if r:
                records.append(r)
        except Exception as e:
            print(f"Warning: Failed to parse {jf.name}: {e}")

    if not records:
        print("No valid records extracted.")
        return

    df = pd.DataFrame(records)

    # Sort primarily by best_accuracy_pct (descending), then by final_test_loss (ascending)
    df = df.sort_values(by=["best_accuracy_pct", "final_test_loss"], ascending=[False, True]).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank

    # Output file path
    output_csv = Path(args.output_csv) if args.output_csv else input_path / "grid_search_summary.csv"
    df.to_csv(output_csv, index_label="rank")

    print("\n" + "=" * 100)
    print(f"🏆 TOP CONFIGURATIONS (Sorted by Best Accuracy):")
    print("=" * 100)
    display_cols = [
        "trial_name", "learning_rate", "local_epochs", "batch_size", "fraction_fit",
        "best_accuracy_pct", "best_accuracy_round", "final_test_loss", "final_coverage_pct"
    ]
    # Filter only available columns
    available_display = [c for c in display_cols if c in df.columns]
    print(df[available_display].head(10).to_string())
    print("=" * 100)
    print(f"💾 Full summary spreadsheet exported to:")
    print(f"   👉 {output_csv.resolve()}\n")


if __name__ == "__main__":
    main()
