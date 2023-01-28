
# -- import packages: ----------------------------------------------------------
from torchdiffeq import odeint


# -- import local dependencies: ------------------------------------------------
from ._base_lightning_diffeq import BaseLightningDiffEq


# -- model class: --------------------------------------------------------------
class BaseLightningODE(BaseLightningDiffEq):
    def __init__(self):
        super(BaseLightningODE, self).__init__()

    def forward(self, X0, t, stage=None, **kwargs):
        """
        We want this to be easily-accesible from the outside, so we
        directly define the forward step with the integrator code.
        """
        
        return odeint(self.func, X0, t=t, **kwargs)
