# _scDiffEq_model_module.py
__module_name__ = "_scDiffEq_model_module.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# package imports #
# --------------- #
# import torch


# local imports #
# ------------- #
from ._supporting_functions._neural_differential_equation import (
    _Neural_Differential_Equation,
)


class _scDiffEq:
    def __init__(self, drift=True, diffusion=True, **kwargs):
        """
        drift=True,
        diffusion=True,
        in_dim=2,
        out_dim=2,
        layers=2,
        nodes=5,
        activation_function=torch.nn.Tanh(),
        batch_size=10,
        brownian_size=1,
        """

        self.model = _Neural_Differential_Equation(drift, diffusion)

    def preflight(self, adata):

        print(self.__dir__())

    def learn(self,):

        print("")

    def evaluate(self,):

        print("")

    def compute_quasi_potential(self,):

        print("")

    def save(self,):

        print("")
