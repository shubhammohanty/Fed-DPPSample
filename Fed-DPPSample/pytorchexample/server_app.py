"""pytorchexample: A Flower / PyTorch app."""

import os
import json
import datetime
from pathlib import Path
import timeit
from logging import INFO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from flwr.common import log
from flwr.server import Server, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.compat.server.serverapp_components import ServerAppComponents
from flwr.compat.common.typing import Parameters
from flwr.serverapp import ServerApp
from flwr.app import Context
from flwr.compat.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.compat.common.typing import FitIns, EvaluateIns
from concurrent.futures import ThreadPoolExecutor
from pytorchexample.task import GRUForecaster, get_weights, load_centralized_dataset, test

_SERVER_DATA_DIR = Path("/driveB/FL/group9/flproject/Fed-DPPSample/pytorchexample/dataset")
DATA_DIR = _SERVER_DATA_DIR if _SERVER_DATA_DIR.exists() else (Path(__file__).resolve().parent / "dataset")

app = ServerApp()


# ── Graph data ────────────────────────────────────────────────────────────────

def load_graph_data(centrality_type="degree"):
    sp = pd.read_csv(
        os.path.join(DATA_DIR, "PeMSD7_Shortest_Paths_228x228.csv"),
        index_col=0
    ).values.astype(np.float32)

    if centrality_type == "degree":
        df = pd.read_csv(os.path.join(DATA_DIR, "PeMSD7_Degree_Centrality.csv"))
        centrality = df["Degree_Centrality"].values.astype(np.float32)
    else:
        df = pd.read_csv(os.path.join(DATA_DIR, "PeMSD7_PageRank_Centrality.csv"))
        centrality = df["PageRank_Centrality"].values.astype(np.float32)

    return sp, centrality


# ── DPP sampling ──────────────────────────────────────────────────────────────

def build_L(centrality, shortest_paths, ages, sigma, alpha):
    q = centrality * (1.0 + alpha * ages)
    S = np.exp(-(shortest_paths ** 2) / (2 * sigma ** 2))
    L = np.outer(q, q) * S
    return L


def dpp_sample(centrality, shortest_paths, ages, k, sigma, alpha):
    from dppy.finite_dpps import FiniteDPP
    L = build_L(centrality, shortest_paths, ages, sigma, alpha).astype(np.float64)
    # Ensure L is symmetric and Positive Semi-Definite (PSD)
    L = (L + L.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(L)
    eigvals = np.clip(eigvals, 1e-7, None)
    # Pass the clipped eigendecomposition directly, so dppy does not
    # re-decompose L and hit tiny negative eigenvalues from round-off
    dpp = FiniteDPP(kernel_type="likelihood", L_eig_dec=(eigvals, eigvecs))
    dpp.sample_exact_k_dpp(size=k)
    return [int(x) for x in dpp.list_of_samples[0]]


def random_sample(k, num_clients):
    return [int(x) for x in np.random.choice(num_clients, size=k, replace=False)]


def poc_sample(k, d, parameters, all_clients, cid_list, server_round):
    """Power-of-Choice (Cho et al.): pick d random candidates, query their
    local training loss on the current global model, keep the top-k."""
    num_clients = len(cid_list)
    d = min(max(d, k), num_clients)
    candidates = [int(x) for x in np.random.choice(num_clients, size=d, replace=False)]
    ins = EvaluateIns(parameters, {"loss-on": "train"})

    def query(i):
        res = all_clients[cid_list[i]].evaluate(ins, timeout=None, group_id=server_round)
        return i, float(res.loss)

    with ThreadPoolExecutor(max_workers=d) as ex:
        losses = dict(ex.map(query, candidates))
    top_k = sorted(candidates, key=lambda i: losses[i], reverse=True)[:k]
    return top_k, candidates, losses


# ── Custom Strategy ───────────────────────────────────────────────────────────

class DPPFedAvg(FedAvg):
    def __init__(self, shortest_paths, centrality, sigma, alpha,
                 sampling_mode, fraction_fit, poc_d=20, **kwargs):
        super().__init__(fraction_fit=fraction_fit, **kwargs)
        self.poc_d = poc_d
        self.poc_log = []  # per-round PoC candidates + losses (extra comm cost)
        self.shortest_paths = shortest_paths
        self.centrality = centrality
        self.sigma = sigma
        self.alpha = alpha
        self.sampling_mode = sampling_mode
        self.ages = None  # init when we know num_clients
        self.all_selected = []  # log for analysis

    def configure_fit(self, server_round, parameters, client_manager):
        """Override client selection here."""

        # Wait until enough clients are available
        client_manager.wait_for(self.min_available_clients)
        all_clients = client_manager.all()  # dict {cid: ClientProxy}

        # Map cid to index (cid is a string like "0", "1", ...)
        cid_list = sorted(all_clients.keys(), key=lambda x: int(x))
        num_clients = len(cid_list)
        k = max(1, int(self.fraction_fit * num_clients))

        # Init ages first time
        if self.ages is None:
            self.ages = np.zeros(num_clients, dtype=np.float32)

        # Sample
        if self.sampling_mode == "dpp":
            selected_idx = dpp_sample(
                self.centrality, self.shortest_paths,
                self.ages, k, self.sigma, self.alpha
            )
        elif self.sampling_mode == "poc":
            selected_idx, cands, losses = poc_sample(
                k, self.poc_d, parameters, all_clients, cid_list, server_round
            )
            self.poc_log.append({"round": server_round, "candidates": cands,
                                 "losses": {str(i): losses[i] for i in cands}})
            print(f"[Round {server_round}] PoC queried {len(cands)} clients for loss")
        else:
            selected_idx = random_sample(k, num_clients)

        # Update ages
        self.ages += 1
        self.ages[selected_idx] = 0

        # Log
        self.all_selected.append(selected_idx)
        print(f"\n[Round {server_round}] Selected {len(selected_idx)} clients: {selected_idx}")

        # Build FitIns for selected clients
        config = {}
        fit_ins = FitIns(parameters, config)
        selected_clients = [(all_clients[cid_list[i]], fit_ins) for i in selected_idx]
        return selected_clients


def get_evaluate_fn(testloader, device, hidden_dim=32):
    def evaluate(server_round: int, parameters, config):
        model = GRUForecaster(hidden_dim=hidden_dim)
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(model.state_dict().keys(), parameters)
        }
        model.load_state_dict(state_dict)
        loss, mae = test(model, testloader, device)
        accuracy = max(0.0, (1.0 - mae)) * 100.0
        print(f"\n[Round {server_round}] Centralized Test Loss: {loss:.4f}, MAE: {mae:.4f}, Accuracy: {accuracy:.2f}%")
        return loss, {"mae": mae, "accuracy": accuracy}
    return evaluate


# ── Results Logging & Plotting ────────────────────────────────────────────────

def save_experiment_results(run_metadata, round_records):
    _SERVER_RESULTS_DIR = Path("/driveB/FL/group9/flproject/Fed-DPPSample/results")
    results_dir = _SERVER_RESULTS_DIR if _SERVER_RESULTS_DIR.parent.exists() else (Path(__file__).resolve().parent.parent / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = run_metadata.get("sampling-mode", "unknown")

    # Save JSON log
    seed = run_metadata.get("seed", "na")
    json_path = results_dir / f"run_{mode}_s{seed}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({"metadata": run_metadata, "history": round_records}, f, indent=2)

    # Generate 4-panel Plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    rounds = [r["round"] for r in round_records]

    # Panel 1: Train Loss vs Test Loss (Overfitting monitor)
    ax1 = axs[0, 0]
    train_losses = [r["train_loss"] for r in round_records if r["train_loss"] is not None]
    train_rounds = [r["round"] for r in round_records if r["train_loss"] is not None]
    test_losses = [r["test_loss"] for r in round_records]
    if train_losses:
        ax1.plot(train_rounds, train_losses, "o--", label="Client Train Loss", color="#1f77b4")
    ax1.plot(rounds, test_losses, "s-", label="Global Test Loss", color="#d62728")
    ax1.set_title("Train Loss vs Test Loss (Overfitting Check)")
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("MSE Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Panel 2: Test Accuracy (%) & MAE
    ax2 = axs[0, 1]
    accuracies = [r["test_accuracy"] for r in round_records]
    maes = [r["test_mae"] for r in round_records]
    color_acc = "#2ca02c"
    ax2.plot(rounds, accuracies, "o-", color=color_acc, label="Accuracy (%)")
    ax2.set_title("Global Test Accuracy & MAE vs Round")
    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Accuracy (%)", color=color_acc)
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.grid(True, linestyle="--", alpha=0.6)

    ax2_twin = ax2.twinx()
    color_mae = "#ff7f0e"
    ax2_twin.plot(rounds, maes, "s--", color=color_mae, label="MAE")
    ax2_twin.set_ylabel("MAE", color=color_mae)
    ax2_twin.tick_params(axis="y", labelcolor=color_mae)

    # Panel 3: Cumulative Sensor Coverage (%)
    ax3 = axs[1, 0]
    coverage = [r["cumulative_coverage_percent"] for r in round_records]
    ax3.plot(rounds, coverage, "^-", color="#9467bd", label="Sensor Coverage (%)")
    ax3.axhline(100.0, color="gray", linestyle=":", label="100% Network Coverage")
    ax3.set_title("Cumulative Network Coverage (%)")
    ax3.set_xlabel("Communication Round")
    ax3.set_ylabel("Unique Sensors Sampled (%)")
    ax3.set_ylim([0, 105])
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="lower right")

    # Panel 4: Spatial Dispersion of Selected Sensors
    ax4 = axs[1, 1]
    dispersions = [r["mean_spatial_distance"] for r in round_records if r["mean_spatial_distance"] is not None]
    disp_rounds = [r["round"] for r in round_records if r["mean_spatial_distance"] is not None]
    if dispersions:
        ax4.plot(disp_rounds, dispersions, "d-", color="#8c564b", label="Mean Pairwise Distance")
    ax4.set_title("Spatial Dispersion (Mean Graph Shortest-Path Distance)")
    ax4.set_xlabel("Communication Round")
    ax4.set_ylabel("Distance (hops / weight)")
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend()

    plt.tight_layout()
    png_path = results_dir / f"run_{mode}_s{seed}_{timestamp}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    log(INFO, f"\n📁 [SAVED] Experiment JSON: {json_path}")
    log(INFO, f"📈 [SAVED] Experiment Plot: {png_path}\n")


# ── Early Stopping Server ─────────────────────────────────────────────────────

class EarlyStoppingServer(Server):
    def __init__(self, target_accuracy: float = 95.0, run_metadata: dict = None, shortest_paths: np.ndarray = None, **kwargs):
        super().__init__(**kwargs)
        self.target_accuracy = target_accuracy
        self.run_metadata = run_metadata or {}
        self.shortest_paths = shortest_paths

    def fit(self, num_rounds: int, timeout: float | None):
        history = History()
        round_records = []

        # Initialize parameters
        log(INFO, "[INIT]")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
        log(INFO, "Starting evaluation of initial global parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
            round_records.append({
                "round": 0,
                "train_loss": None,
                "test_loss": float(res[0]),
                "test_mae": float(res[1]["mae"]),
                "test_accuracy": float(res[1]["accuracy"]),
                "selected_clients": [],
                "mean_spatial_distance": None,
                "cumulative_sensors_covered": 0,
                "cumulative_coverage_percent": 0.0,
            })
        else:
            log(INFO, "Evaluation returned no results (`None`)")

        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)

            train_loss = None
            if res_fit is not None:
                parameters_prime, fit_metrics, results_and_failures = res_fit
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )
                # Compute average local client training loss
                if results_and_failures and results_and_failures[0]:
                    losses = [
                        fit_res.metrics["train_loss"]
                        for _, fit_res in results_and_failures[0]
                        if "train_loss" in fit_res.metrics
                    ]
                    if losses:
                        train_loss = float(np.mean(losses))

            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

                # Compute spatial metrics
                selected_idx = self.strategy.all_selected[-1] if self.strategy.all_selected else []
                mean_dist = None
                if len(selected_idx) > 1 and self.shortest_paths is not None:
                    dists = [
                        float(self.shortest_paths[i, j])
                        for i in selected_idx
                        for j in selected_idx
                        if i < j
                    ]
                    mean_dist = float(np.mean(dists)) if dists else 0.0

                all_seen = set(idx for round_sel in self.strategy.all_selected for idx in round_sel)
                unique_count = len(all_seen)
                coverage_pct = round((unique_count / 228.0) * 100.0, 2)

                round_records.append({
                    "round": current_round,
                    "train_loss": train_loss,
                    "test_loss": float(loss_cen),
                    "test_mae": float(metrics_cen["mae"]),
                    "test_accuracy": float(metrics_cen["accuracy"]),
                    "selected_clients": [int(x) for x in selected_idx],
                    "mean_spatial_distance": mean_dist,
                    "cumulative_sensors_covered": unique_count,
                    "cumulative_coverage_percent": coverage_pct,
                    "poc_queried": len(self.strategy.poc_log[-1]["candidates"])
                        if getattr(self.strategy, "poc_log", None) and self.strategy.sampling_mode == "poc" else 0,
                })

                acc = metrics_cen.get("accuracy", 0.0)
                if acc >= self.target_accuracy:
                    log(INFO, f"\n🎯 Target accuracy of {self.target_accuracy}% reached at Round {current_round}! (Accuracy: {acc:.2f}%). Stopping early.")
                    break

        end_time = timeit.default_timer()
        elapsed = end_time - start_time

        # Save run results and generate plots
        try:
            save_experiment_results(self.run_metadata, round_records)
        except Exception as e:
            log(INFO, f"Warning: Failed to save results plot: {e}")

        return history, elapsed


# ── server_fn ─────────────────────────────────────────────────────────────────

def server_fn(context: Context) -> ServerAppComponents:

    num_rounds      = context.run_config["num-server-rounds"]
    fraction_fit    = context.run_config["fraction-fit"]
    sigma           = context.run_config["sigma"]
    alpha           = context.run_config["alpha"]
    centrality_type = context.run_config["centrality-type"]
    sampling_mode   = context.run_config["sampling-mode"]
    target_accuracy = context.run_config.get("target-accuracy", 95.0)
    hidden_dim      = context.run_config.get("hidden-dim", 32)
    poc_d           = context.run_config.get("poc-d", 20)
    seed            = context.run_config.get("seed", 42)

    np.random.seed(seed)
    torch.manual_seed(seed)

    shortest_paths, centrality = load_graph_data(centrality_type)

    # Init global model and get initial parameters
    model = GRUForecaster(hidden_dim=hidden_dim)
    initial_parameters = ndarrays_to_parameters(get_weights(model))

    device = torch.device("cpu")
    testloader = load_centralized_dataset(batch_size=64)
    evaluate_fn = get_evaluate_fn(testloader, device, hidden_dim=hidden_dim)

    strategy = DPPFedAvg(
        shortest_paths=shortest_paths,
        centrality=centrality,
        sigma=sigma,
        alpha=alpha,
        sampling_mode=sampling_mode,
        fraction_fit=fraction_fit,
        poc_d=poc_d,
        fraction_evaluate=0.0,
        initial_parameters=initial_parameters,
        min_available_clients=228,
        evaluate_fn=evaluate_fn,
    )

    client_manager = SimpleClientManager()
    run_metadata = dict(context.run_config)
    server = EarlyStoppingServer(
        client_manager=client_manager,
        strategy=strategy,
        target_accuracy=target_accuracy,
        run_metadata=run_metadata,
        shortest_paths=shortest_paths,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(server=server, strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
