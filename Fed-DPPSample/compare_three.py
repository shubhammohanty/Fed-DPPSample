#!/usr/bin/env python3
"""Compare Random vs DPP vs Power-of-Choice (PoC) across seeds.
Usage (inside Fed-DPPSample):  python compare_three.py
Outputs: results/summary_three.csv  and  results/compare_three.png
"""
import json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("results")
MODES = ["random", "dpp", "poc"]
LABEL = {"random": "FedAvg (Random)", "dpp": "DPP (Ours)", "poc": "Power-of-Choice"}
COLOR = {"random": "#d62728", "dpp": "#1f77b4", "poc": "#2ca02c"}
THRESH = [85.0, 90.0, 93.0]

# Keep the latest file per (mode, seed); ignore short test runs (< 20 rounds)
runs = {}
for f in sorted(RES.glob("run_*_s*_*.json")):
    m = re.match(r"run_(\w+?)_s(\d+)_", f.name)
    if not m or m.group(1) not in MODES:
        continue
    hist = json.load(open(f))["history"]
    if len(hist) < 21:
        continue
    runs[(m.group(1), int(m.group(2)))] = hist

def rounds_to(h, t):
    for r in h[1:]:
        if r["test_accuracy"] >= t:
            return r["round"]
    return np.nan

rows = []
for mode in MODES:
    hs = [h for (m, s), h in sorted(runs.items()) if m == mode]
    if not hs:
        print(f"No runs found for {mode}")
        continue
    stat = lambda vals: (np.nanmean(vals), np.nanstd(vals))
    last10 = [np.mean([r["test_accuracy"] for r in h[-10:]]) for h in hs]
    peak = [max(r["test_accuracy"] for r in h[1:]) for h in hs]
    minloss = [min(r["test_loss"] for r in h[1:]) for h in hs]
    cov = [h[-1]["cumulative_coverage_percent"] for h in hs]
    disp = [np.mean([r["mean_spatial_distance"] for r in h[1:] if r["mean_spatial_distance"] is not None]) for h in hs]
    extra = [sum(r.get("poc_queried", 0) for r in h) for h in hs]
    row = {"method": LABEL[mode], "seeds": len(hs),
           "final_acc(last10)": stat(last10), "peak_acc": stat(peak),
           "min_test_mse": stat(minloss), "coverage_%": stat(cov),
           "mean_spatial_dist": stat(disp), "extra_loss_queries": stat(extra)}
    for t in THRESH:
        row[f"rounds_to_{t:.0f}%"] = stat([rounds_to(h, t) for h in hs])
    rows.append(row)

# Print table
keys = [k for k in rows[0] if k not in ("method", "seeds")]
print(f"\n{'metric':22s}" + "".join(f"{r['method']+' (n='+str(r['seeds'])+')':>28s}" for r in rows))
for k in keys:
    print(f"{k:22s}" + "".join(f"{r[k][0]:>18.3f} ± {r[k][1]:<7.3f}" for r in rows))

# Save CSV
with open(RES / "summary_three.csv", "w") as f:
    f.write("metric," + ",".join(r["method"] for r in rows) + "\n")
    for k in keys:
        f.write(k + "," + ",".join(f"{r[k][0]:.4f} ± {r[k][1]:.4f}" for r in rows) + "\n")

# Plot mean ± std curves
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
panels = [("test_accuracy", "Test Accuracy (%)"),
          ("cumulative_coverage_percent", "Cumulative Coverage (%)"),
          ("mean_spatial_distance", "Mean Pairwise Distance")]
for mode in MODES:
    hs = [h for (m, s), h in runs.items() if m == mode]
    if not hs:
        continue
    n = min(len(h) for h in hs)
    x = [r["round"] for r in hs[0][1:n]]
    for ax, (key, title) in zip(axs, panels):
        Y = np.array([[r[key] if r[key] is not None else np.nan for r in h[1:n]] for h in hs], dtype=float)
        mu, sd = np.nanmean(Y, 0), np.nanstd(Y, 0)
        ax.plot(x, mu, color=COLOR[mode], label=LABEL[mode], lw=2)
        ax.fill_between(x, mu - sd, mu + sd, color=COLOR[mode], alpha=0.2)
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("Communication Round")
        ax.grid(True, ls="--", alpha=0.6)
axs[0].set_ylim(80, 96)
for ax in axs:
    ax.legend()
plt.tight_layout()
fig.savefig(RES / "compare_three.png", dpi=300)
print(f"\nSaved: {RES/'summary_three.csv'}  and  {RES/'compare_three.png'}")
