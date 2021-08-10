
# package imports #
# --------------- #
import torch
import vintools as v
import torch.nn as nn
from collections import OrderedDict

# local imports #
# ------------- #
from ._check_pytorch_activation_function import _check_pytorch_activation_function

def _create_flexible_neural_network(
    in_dim=2,
    out_dim=2,
    n_layers=3,
    nodes_n=15,
    nodes_m=20,
    activation_function=nn.Tanh(),
):

    """

    Parameters:
    -----------
    in_dim
        default: 2
    out_dim
        default: 2
    n_layers
        default: 3
    nodes
        default: 15
    activation_function
        default: nn.Tanh()
        type: torch.nn.modules.activation.Tanh

    Returns:
    --------
    nn.Sequential(neural_network)
    """

    _check_pytorch_activation_function(activation_function)
    neural_network = OrderedDict()

    for layer in range(n_layers):
        if layer == 0:
            neural_network[str(layer)] = nn.Linear(in_dim, nodes_n)
        elif layer == (n_layers - 1):
            neural_network[str(layer)] = nn.Linear(nodes_m, out_dim)
        else:
            if layer % 2 == 0:
                neural_network[str(layer)] = nn.Linear(nodes_m, nodes_n)
            else:
                neural_network[str(layer)] = nn.Linear(nodes_n, nodes_m)

        if layer != (n_layers - 1):
            neural_network["af_" + str(layer)] = activation_function

    return nn.Sequential(neural_network)