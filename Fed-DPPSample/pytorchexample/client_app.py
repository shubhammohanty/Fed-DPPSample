"""pytorchexample: A Flower / PyTorch app."""

import torch
import numpy as np
from flwr.clientapp import ClientApp
from flwr.compat.client.numpy_client import NumPyClient
from flwr.app import Context

from pytorchexample.task import GRUForecaster, load_data, train, test, get_weights


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    batch_size   = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    local_steps  = context.run_config.get("local-steps", 20)
    lr           = context.run_config["learning-rate"]
    hidden_dim   = context.run_config.get("hidden-dim", 32)
    return FlowerClient(partition_id, batch_size, local_epochs, local_steps, lr, hidden_dim).to_client()


class FlowerClient(NumPyClient):
    def __init__(self, partition_id, batch_size, local_epochs, local_steps, lr, hidden_dim):
        self.partition_id  = partition_id
        self.batch_size    = batch_size
        self.local_epochs  = local_epochs
        self.local_steps   = local_steps
        self.lr            = lr
        self.device        = torch.device("cpu")
        self.model         = GRUForecaster(hidden_dim=hidden_dim).to(self.device)
        self.trainloader, self.valloader = load_data(partition_id, batch_size)

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        # Load global weights into model
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict)

        # Train
        train_loss = train(self.model, self.trainloader, self.local_epochs, self.lr, self.device, self.local_steps)
        print(f"[Client {self.partition_id}] Train loss: {train_loss:.4f}")
        return get_weights(self.model), len(self.trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict)
        # PoC asks for local *training* loss; normal eval uses validation data
        if config.get("loss-on") == "train":
            loss, mae = test(self.model, self.trainloader, self.device)
            return loss, len(self.trainloader.dataset), {"mae": mae}
        loss, mae = test(self.model, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"mae": mae}


app = ClientApp(client_fn=client_fn)
