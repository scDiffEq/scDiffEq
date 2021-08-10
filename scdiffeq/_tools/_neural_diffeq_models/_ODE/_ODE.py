
# package imports #
# --------------- #
import torch.nn as nn

# local imports #
# ------------- #
from ._supporting_functions._create_flexible_neural_network import _create_flexible_neural_network
from ._supporting_functions._learn_ODE import learn

class Neural_ODE(nn.Module):

    """
    Instantiates a neural network, which serves as an ODE for use with neural ODEs.

    Parameters:
    -----------
    in_dim
        Number of nodes in neural network at data entry point.
        default: 2
        type: int
    
    out_dim
        Number of nodes in neural network at exit point.
        default: 2
        type: int
    
    n_layers
        Number of layers in neural network.
        default: 2
        type: int
    
    nodes_n
        Number of nodes in neural network layers.
        default: 5
        type: int
    
    nodes_m
        Number of nodes in neural network layers.
        default: 5
        type: int
    
    activation_function
        Torch activation function placed between layers.
            default: nn.Tanh()
            type: <class 'torch.nn.modules.activation.[function]'>
    
    
    Returns:
    --------
    _Neural_ODE(
      (net): Sequential(
        (0): Linear(in_features=in_dim, out_features=nodes_m, bias=True)
        (af_0): activation_function()
        (1): Linear(in_features=nodes_m, out_features=nodes_n, bias=True)
        (af_1): activation_function()
        [...]
        (n-1): Linear(in_features=nodes_n, out_features=nodes_m, bias=True)
        (af_n-1): activation_function()
        (n): Linear(in_features=nodes_m, out_features=out_dim, bias=True)
      )
    )
    """

    def __init__(
        self,
        in_dim=2,
        out_dim=2,
        n_layers=5,
        nodes_n=50,
        nodes_m=50,
        activation_function=nn.Tanh(),
    ):
        super(Neural_ODE, self).__init__()

        self.net = _create_flexible_neural_network(
            in_dim=in_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            nodes_n=nodes_n,
            nodes_m=nodes_m,
            activation_function=activation_function,
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)