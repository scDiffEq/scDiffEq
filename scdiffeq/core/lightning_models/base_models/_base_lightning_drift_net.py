
# -- import packages: ----------------------------------------------------------
import torch


# -- import local dependencies: ------------------------------------------------
from ._base_lightning_diffeq import BaseLightningDiffEq


# -- type setting: -------------------------------------------------------------
NoneType = type(None)


# -- base sub-classes: ---------------------------------------------------------
class BaseLightningDriftNet(BaseLightningDiffEq):
    def __init__(self, dt=0.1, stdev=torch.Tensor([0.5])):
        super(BaseLightningDriftNet, self).__init__()
        
        from brownian_diffuser import nn_int
        self.nn_int = nn_int
            
    def integrate(self, X0, t, stage=None, max_steps=None, return_all=False):
        return self.nn_int(
            self.func,
            X0=X0,
            t=t,
            dt=self.hparams["dt"],
            stdev=self.hparams["stdev"],
            max_steps=max_steps,
            return_all=return_all,
        )
