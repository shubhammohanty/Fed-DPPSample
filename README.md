# Topology-Aware Determinantal Client Sampling (Fed-DPPSample)

This repository provides the official implementation of **Topology-Aware Determinantal Client Sampling** for federated spatio-temporal sensor networks on the PeMSD7-M benchmark ($N=228$ clients). 

The codebase implements and compares three client selection strategies under a unified federated framework built on Flower and PyTorch:
1. **Topology-Aware DPP Sampling**: Combines graph shortest-path geodesic distance kernels, topological centrality metrics (Degree or PageRank), and dynamic staleness/age penalization for participant selection without requiring client-side data or gradient uploads prior to selection.
2. **Power-of-Choice (PoC)**: Active selection baseline (Cho et al.) that queries candidate clients for their local training loss on the global model and samples the highest-loss devices.
3. **Uniform Random Sampling**: Standard FedAvg baseline (McMahan et al.) with uniform sampling without replacement.

---

## 1. Repository Structure

```text
Fed-DPPSample/
├── pytorchexample/
│   ├── __init__.py
│   ├── client_app.py           # Flower ClientApp (training, validation, PoC loss queries)
│   ├── server_app.py           # ServerApp with DPPFedAvg strategy and early-stopping server
│   ├── task.py                 # GRUForecaster model architecture, dataset loader, and metrics
│   └── dataset/                # PeMSD7-M traffic speed series and graph topology files
│       ├── distance.py
│       ├── PeMSD7_Degree_Centrality.csv
│       ├── PeMSD7_PageRank_Centrality.csv
│       ├── PeMSD7_Shortest_Paths_228x228.csv
│       ├── pemsd7_train.csv
│       ├── pemsd7_val.csv
│       └── pemsd7_test.csv
├── results/                    # Output directory for experiment JSON logs and trajectory figures
├── pyproject.toml              # Flower federation settings and default hyperparameters
├── requirements.txt            # Python package dependencies
├── setup_env.sh                # Automated Python virtualenv setup script
├── setup_env_conda.sh          # Automated Conda environment setup script
├── run_stage2_dpp_suite.py     # Parameter sensitivity and ablation study runner (sigma, alpha, centrality)
├── run_stage2_dpp.sh           # SLURM submission script for Stage 2 suite
├── run_poc100.sh               # SLURM submission script for 100-round Power-of-Choice experiment
├── test_poc.sh                 # Fast 3-round Power-of-Choice sanity check
├── run_grid_stage1.sh          # SLURM script for Stage 1 baseline hyperparameter grid search
├── grid_search_stage1_random.py# Stage 1 grid search runner
├── export_grid_summary.py      # Results aggregation utility for Stage 1 grid search
├── compare_three.py            # Comparative evaluation script (aggregates Random, PoC, and DPP runs)
├── compare.sh                  # SLURM submission script for comparative evaluation
└── plot_paper_suite.py         # Figure generation script for trajectory and sensitivity plots
```

---

## 2. Environment Setup

### System Prerequisites
- Operating System: Linux (tested on Ubuntu 22.04 LTS and Arch Linux)
- Python: 3.10 to 3.12
- Hardware: CUDA-compatible GPU recommended; CPU execution supported

### Option A: Automated Virtual Environment (Recommended)
```bash
git clone https://github.com/shubhammohanty/Fed-DPPSample.git
cd Fed-DPPSample
bash setup_env.sh
source .venv/bin/activate
```

### Option B: Automated Conda Environment
```bash
git clone https://github.com/shubhammohanty/Fed-DPPSample.git
cd Fed-DPPSample
bash setup_env_conda.sh
conda activate dpp_fl
```

### Option C: Manual Installation
```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 3. Configure Flower simulation engine for 228 SuperNodes
flwr federation simulation-config @none/default --num-supernodes 228 --client-resources-num-gpus 0.5
```

---

## 3. Configuration

Hyperparameters and sampling configurations are defined in `pyproject.toml` under `[tool.flwr.app.config]`.

```toml
[tool.flwr.app.config]
num-server-rounds = 100
learning-rate = 0.0005
fraction-fit = 0.04            # Cohort size: 9 clients out of 228 (k = 9)
sigma = 2.0                    # Spatial Gaussian kernel bandwidth (km)
alpha = 0.5                    # Staleness / age penalty weight
centrality-type = "degree"     # "degree" or "pagerank"
sampling-mode = "dpp"          # "dpp", "poc", or "random"
poc-d = 20                     # Candidate pool size for Power-of-Choice
seed = 42                      # Random seed for reproducible execution
save-model = false
local-epochs = 2
local-steps = 10
hidden-dim = 16
batch-size = 32
target-accuracy = 95.0
```

Any configuration parameter can also be overridden from the command line:
```bash
flwr run . --run-config "sampling-mode='poc' poc-d=20 seed=42"
```

---

## 4. Running Experiments

### A. Executing Individual Client Selection Strategies

#### 1. Topology-Aware DPP Sampling
Executes exact $k$-DPP sampling using graph geodesic distances and topological quality scores:
```bash
flwr run . --run-config "sampling-mode='dpp' sigma=2.0 alpha=0.5 centrality-type='degree' num-server-rounds=100 seed=42"
```

#### 2. Power-of-Choice Baseline
Executes active loss-greedy selection from a candidate pool of size $d=20$:
```bash
flwr run . --run-config "sampling-mode='poc' poc-d=20 num-server-rounds=100 seed=42"
```

#### 3. Uniform Random Baseline (FedAvg)
Executes uniform random selection without replacement:
```bash
flwr run . --run-config "sampling-mode='random' num-server-rounds=100 seed=42"
```

**Experiment Outputs:**  
Each run automatically logs metrics to `results/`:
- `results/run_<mode>_s<seed>_<timestamp>.json`: Per-round centralized accuracy, MSE loss, MAE, mean spatial cohort distance, cumulative network coverage, and selected client indices.
- `results/run_<mode>_s<seed>_<timestamp>.png`: 4-panel diagnostic plot of training loss, test accuracy, spatial dispersion, and cumulative sensor coverage over time.

---

### B. Stage 2: DPP Parameter Sensitivity and Ablation Study

Runs the complete systematic parameter sweep:
- Spatial bandwidths: $\sigma \in \{1.0, 2.0, 5.0\}$
- Staleness penalties: $\alpha \in \{0.1, 0.5, 1.0\}$
- Graph centrality metrics: Degree vs. PageRank

**Execution:**
- Local execution:
  ```bash
  python run_stage2_dpp_suite.py
  ```
- SLURM cluster execution:
  ```bash
  sbatch run_stage2_dpp.sh
  ```

Output files are stored under `results/paper_experiments/`.

---

### C. Stage 1: Baseline Hyperparameter Grid Search

Performs the initial grid search over learning rates, batch sizes, and local steps for the GRU forecaster under standard FedAvg.

**Execution:**
- Local execution:
  ```bash
  python grid_search_stage1_random.py
  ```
- SLURM cluster execution:
  ```bash
  sbatch run_grid_stage1.sh
  ```
- Tabulate and export summary statistics:
  ```bash
  python export_grid_summary.py --results-dir results/grid_search_stage3
  ```

---

### D. Comparative Evaluation and Metrics Summary

To aggregate completed runs across Random, DPP, and Power-of-Choice:
```bash
python compare_three.py
```

This generates:
- `results/summary_three.csv`: Comparative table of rounds to reach accuracy thresholds (85%, 90%), peak accuracy, final test MSE, network coverage percentage, and communication query overhead.
- `results/compare_three.png`: 3-panel publication comparison figure.

---

### E. Generating Paper Figures

To re-render all publication figures from saved experiment JSON records:
```bash
python plot_paper_suite.py --results-dir results/paper_experiments
```

---

## 5. SLURM Cluster Workload Management

Batch submission scripts are provided for distributed HPC execution:
- `sbatch run_stage2_dpp.sh`: Executes the full 7-configuration Stage 2 sensitivity suite.
- `sbatch run_poc100.sh`: Submits a 100-round Power-of-Choice run ($d=20$).
- `sbatch test_poc.sh`: Submits a short 3-round test job to verify setup.
- `sbatch compare.sh`: Aggregates cluster runs and compiles summary tables.
- `sbatch run_grid_stage1.sh`: Submits the Stage 1 baseline hyperparameter search.

Job stdout and stderr files are directed to `logs/`.
