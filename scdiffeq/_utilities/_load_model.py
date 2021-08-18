import torch
import torch.nn as nn
from ._torch_device import _set_device


def _specify_func(data_dimensionality, layers, nodes):

    """Currently this function is only for a 1-layer network."""

    class ODEFunc(nn.Module):
        def __init__(self, data_dimensionality, layers, nodes):
            super(ODEFunc, self).__init__()

            if layers == 1:
                self.net = nn.Sequential(
                    nn.Linear(data_dimensionality, nodes),
                    nn.Tanh(),
                    nn.Linear(nodes, data_dimensionality),
                )
            if layers == 2:
                self.net = nn.Sequential(
                    nn.Linear(data_dimensionality, nodes),
                    nn.Tanh(),
                    nn.Linear(nodes, data_dimensionality),
                    nn.Tanh(),
                    nn.Linear(data_dimensionality, nodes),
                    nn.Tanh(),
                    nn.Linear(nodes, data_dimensionality),
                )
            if layers == 3:
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
            if layers == 4:
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

            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.1)
                    nn.init.constant_(m.bias, val=0)

        def forward(self, t, y):
            return self.net(y)

    func = ODEFunc(data_dimensionality, layers, nodes).to(_set_device())

    return func


def _load_model(path, data_dimensionality, layers, nodes):

    """
    Load a saved ODEFunc model. 
    
    Parameters:
    -----------
    model
    
    path
        path to saved model
    
    data_dimensionality
    
    layers
        number of layers in the saved model
    
    nodes
        number of nodes in the saved model
    
    Returns:
    --------
    model
        saved model
    
    """

    model = _specify_func(data_dimensionality, layers, nodes)

    model.load_state_dict(torch.load(path))
    print(model.eval())

    return model
