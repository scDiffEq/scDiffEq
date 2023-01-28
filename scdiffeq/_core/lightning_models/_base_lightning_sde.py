
# -- import packages: ----------------------------------------------------------
from torchsde import sdeint


# -- import local dependencies: ------------------------------------------------
from ._base_lightning_diffeq import BaseLightningDiffEq


# -- model class: --------------------------------------------------------------
class BaseLightningSDE(BaseLightningDiffEq):
    def __init__(self, dt=0.1):
        super(BaseLightningSDE, self).__init__()

    def forward(self, X0, t, stage=None, **kwargs):
        """
        We want this to be easily-accesible from the outside, so we
        directly define the forward step with the integrator code.
        """
        
        return sdeint(self.func, X0, ts=t, dt=self.hparams["dt"], **kwargs)
