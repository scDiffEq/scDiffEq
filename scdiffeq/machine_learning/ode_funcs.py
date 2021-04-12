### ode_funcs.py
# options of 1-4 layers

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
        else:
            print("Please define ODEFunc layers")
            return 
            

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)  