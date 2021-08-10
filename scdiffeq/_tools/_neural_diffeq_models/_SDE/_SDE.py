
import torch
import torch.nn as nn

class Neural_SDE(nn.Module):
           
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, batch_size, data_dimensionality, brownian_size, nodes=10):
        super().__init__()

        self.batch_size = batch_size
        self.data_dimensionality = data_dimensionality
        self.brownian_size = brownian_size

        self.drift_param = nn.Parameter(
            torch.ones(self.batch_size, self.data_dimensionality) * 0.8, requires_grad=True
        )
        self.diff_constant = torch.ones(self.batch_size, self.data_dimensionality) * 0.000005
        self.diff_constant2 = torch.ones(self.batch_size, self.data_dimensionality) * 0.08

        self.position_net_drift = torch.nn.Sequential(
            nn.Linear(data_dimensionality, nodes),
            nn.ReLU(),
            nn.Linear(nodes, data_dimensionality),
            nn.ReLU(),
            nn.Linear(data_dimensionality, nodes),
            nn.ReLU(),
            nn.Linear(nodes, data_dimensionality),
            nn.ReLU(),
            nn.Linear(data_dimensionality, nodes),
            nn.ReLU(),
            nn.Linear(nodes, data_dimensionality),
        )

        self.position_net_diff = torch.nn.Sequential(nn.Linear(data_dimensionality, data_dimensionality))

    # Drift Function
    def f(self, t, y):
        return self.position_net_drift(y) + self.drift_param

    # Diffusion Function
    def g(self, t, y):
        return (
            self.diff_constant2
            + self.diff_constant * t
            + self.position_net_diff(y) * 0.0001
        )