
# -- import packages: ----------------------------------------------------------
from brownian_diffuser import nn_int
import torch


# -- import local dependencies: ------------------------------------------------
from ._base_lightning_diffeq import BaseLightningDiffEq


# -- model class: --------------------------------------------------------------
class BaseLightningDriftNet(BaseLightningDiffEq):
    def __init__(self, dt=0.1, stdev=torch.Tensor([0.5])):
        super(BaseLightningDriftNet, self).__init__()

        self.save_hyperparameters()

    def forward(self, X0, t, stage=None, max_steps=None, return_all=False):
        return nn_int(
            self.func,
            X0=X0,
            t=t,
            dt=self.hparams["dt"],
            stdev=self.hparams["stdev"],
            max_steps=max_steps,
            return_all=return_all,
        )