
__module_name__ = "Neural_Differential_Equation.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import flexinet
import torch


class Neural_Differential_Equation(torch.nn.Module):

    noise_type = "general"
    sde_type = "ito"

    def __init__(
        self,
        diffusion=True,
        in_dim=25,
        out_dim=25,
        drift_hidden_architecture={1: [500, 500], 2: [500, 500]},
        drift_activation_function=torch.nn.Tanh(),
        diffusion_hidden_architecture={1: [500, 500], 2: [500, 500]},
        diffusion_activation_function=torch.nn.Tanh(),
        drift_dropout=0.1,
        diffusion_dropout=0.1,
        batch_size=1,
        brownian_size=1,
    ):

        super().__init__()

        self._in_dim = in_dim
        self._out_dim = out_dim
        self._batch_size = batch_size
        self._brownian_size = brownian_size
        self._diffusion = diffusion

        if type(drift_dropout) == float:
            drift_dropout_prob = drift_dropout
            drift_dropout = True

        if type(diffusion_dropout) == float:
            diffusion_dropout_prob = diffusion_dropout
            diffusion_dropout = True

        self._drift_network = flexinet.models.compose_nn_sequential(
            in_dim=self._in_dim,
            out_dim=self._out_dim,
            hidden_layer_nodes=drift_hidden_architecture,
            activation_function=drift_activation_function,
            dropout=drift_dropout,
            dropout_probability=drift_dropout_prob,
        )
        if self._diffusion:
            self._diffusion_network = flexinet.models.compose_nn_sequential(
                in_dim=self._in_dim,
                out_dim=self._out_dim,
                hidden_layer_nodes=diffusion_hidden_architecture,
                activation_function=diffusion_activation_function,
                dropout=diffusion_dropout,
                dropout_probability=diffusion_dropout_prob,
            )

    def f(self, t, y):
        return self._drift_network(y)

    def g(self, t, y):
        if self._diffusion:
            self._batch_size = y.shape[0]
            return self._diffusion_network(y).view(
                self._batch_size, self._in_dim, self._brownian_size
            )