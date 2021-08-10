
# package imports #
# --------------- #
import torch.nn as nn

# local imports #
# ------------- #
from ._ODE._ODE import Neural_ODE
from ._SDE._SDE import Neural_SDE
# from .._machine_learning._learn_scDiffEq import _learn_scDiffEq

class scDiffEq:
    def __init__(
        self,
        network_type="ODE",
        in_dim=2,
        out_dim=2,
        n_layers=4,
        nodes_n=50,
        nodes_m=50,
        activation_function=nn.Tanh(),
        silent=False,
    ):

        """
        """
        
        self.available_network_types = ["SDE", "ODE"]

        assert network_type in self.available_network_types, print(
            "Choose from available network types: {}".format(
                self.available_network_types
            )
        )
        if network_type == "ODE":
            self.network = Neural_ODE(
                in_dim=in_dim,
                out_dim=out_dim,
                n_layers=n_layers,
                nodes_n=nodes_n,
                nodes_m=nodes_m,
                activation_function=activation_function,
            )
            
        if not silent:
            print(self.network)
        