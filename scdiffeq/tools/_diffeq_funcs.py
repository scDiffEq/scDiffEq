### ode_funcs.py
# options of 1-4, 25 layers
import torch
import torch.nn as nn

class ode_func(nn.Module):
    def __init__(self, data_dimensionality, layers=1, nodes=5):
        super(ode_func, self).__init__()
        
        if layers == 1:
            self.net = nn.Sequential(
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
            )
        elif layers == 2:
            self.net = nn.Sequential(
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
            )
        elif layers == 3:
            self.net = nn.Sequential(
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
            )
        elif layers == 4:
            self.net = nn.Sequential(
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
            )
            
        elif layers == 25:
            self.net = nn.Sequential(
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
                nn.Tanh(),
                nn.Linear(data_dimensionality, nodes),
                nn.Tanh(),
                nn.Linear(nodes, data_dimensionality),
            )
        else:
            print("Please define ODEFunc layers")
            return 
            

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)  
    
    
class sde_func(torch.nn.Module):
       
    # just start with fixed # of layers for now
    
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