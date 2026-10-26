"""pytorchexample: A Flower / PyTorch app."""

import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os

# ── Model ────────────────────────────────────────────────────────────────────

class GRUForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2,
                 horizon=12, output_dim=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.output_dim = output_dim
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, horizon * output_dim)

    def forward(self, x):
        B, T, N, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        _, hidden = self.gru(x)
        hidden = hidden[-1]
        out = self.head(hidden)
        out = out.reshape(B, N, self.horizon, self.output_dim)
        return out.permute(0, 2, 1, 3)


# ── Dataset ──────────────────────────────────────────────────────────────────

class TrafficDataset(Dataset):
    def __init__(self, data, sensor_idx, T=12, H=12):
        """
        data       : numpy array [total_timesteps, 228], already normalized
        sensor_idx : which sensor (column) this client owns
        T          : input window
        H          : forecast horizon
        """
        self.T = T
        self.H = H
        # grab only this client's sensor column → [total_timesteps, 1]
        self.series = data[:, sensor_idx:sensor_idx+1]
        self.samples = []
        for i in range(len(self.series) - T - H + 1):
            x = self.series[i      : i + T]        # [T, 1]
            y = self.series[i + T  : i + T + H]    # [H, 1]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # model expects [B, T, N, F] so N=1 here
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # [T, 1, 1]
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # [H, 1, 1]
        return x, y


# ── Data loading ─────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent / "dataset"

def _load_csv(fname):
    path = os.path.join(DATA_DIR, fname)
    return pd.read_csv(path, header=0).values.astype(np.float32)  # [rows, 228]

def _get_scaler():
    """Fit min-max scaler on train set only. Returns (min, max) per sensor."""
    train_data = _load_csv("pemsd7_train.csv")
    mn = train_data.min(axis=0)   # [228]
    mx = train_data.max(axis=0)   # [228]
    return mn, mx

def _normalize(data, mn, mx):
    return (data - mn) / (mx - mn + 1e-8)

def load_data(partition_id: int, batch_size: int):
    """
    partition_id = sensor index (0 to 227).
    Returns trainloader, valloader.
    """
    mn, mx = _get_scaler()

    train_raw = _normalize(_load_csv("pemsd7_train.csv"), mn, mx)
    val_raw   = _normalize(_load_csv("pemsd7_val.csv"),   mn, mx)

    train_ds = TrafficDataset(train_raw, sensor_idx=partition_id)
    val_ds   = TrafficDataset(val_raw,   sensor_idx=partition_id)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valloader   = DataLoader(val_ds,   batch_size=batch_size)
    return trainloader, valloader

def load_centralized_dataset(batch_size: int = 64):
    """Full test set across all 228 sensors for server-side eval."""
    mn, mx = _get_scaler()
    test_raw = _normalize(_load_csv("pemsd7_test.csv"), mn, mx)

    # Stack all sensors into one dataset
    all_x, all_y = [], []
    T, H = 12, 12
    for i in range(len(test_raw) - T - H + 1):
        x = test_raw[i     : i + T]       # [T, 228]
        y = test_raw[i + T : i + T + H]   # [H, 228]
        all_x.append(x)
        all_y.append(y)

    # [num_samples, T, 228, 1]
    X = torch.tensor(np.array(all_x), dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(np.array(all_y), dtype=torch.float32).unsqueeze(-1)

    dataset = torch.utils.data.TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size)


# ── Train / Test ─────────────────────────────────────────────────────────────

def train(net, trainloader, epochs, lr, device, local_steps=None):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    net.train()
    running_loss = 0.0
    step_count = 0
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step_count += 1
            if local_steps is not None and step_count >= local_steps:
                return running_loss / step_count
    return running_loss / max(1, step_count)

def test(net, testloader, device):
    net.to(device)
    net.eval()
    criterion = nn.MSELoss()
    mae, mse_loss, count = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            pred = net(x)
            mse_loss += criterion(pred, y).item()
            mae += torch.abs(pred - y).sum().item()
            count += y.numel()
    return mse_loss / len(testloader), mae / count



def get_weights(model):
    """Return model weights as list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
