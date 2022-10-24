
__module_name__ = "_integrators.py"
__doc__ = """Case-specific module for various integrators"""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# -- import packages: --------------------------------------------------------------------
import torch
import torchsde
import torchdiffeq


# -- import local dependencies: ----------------------------------------------------------
from ._base_forward_integrators import BaseForwardIntegrator
from ._brownian_diffuser import BrownianDiffuser


# -- SDE: --------------------------------------------------------------------------------
class SDEIntegrator(BaseForwardIntegrator):
    def __init__(self, dt=0.1, module="sdeint"):
        self._dt = dt
        self._specify_forward_module(torchsde, module)


# -- supporting functions: ODE: ----------------------------------------------------------
def min_max_norm(x, scaler=1):
    return (x - x.min()) / (x.max() - x.min()) * scaler


# -- ODE: --------------------------------------------------------------------------------
class ODEIntegrator(BaseForwardIntegrator):
    def __init__(self, module="odeint"):
        self._specify_forward_module(torchdiffeq, module)

    def _scale_time(self, scaler=1):
        self.kwargs["t"] = min_max_norm(self.kwargs["t"], scaler)

    def _config_optionals(self, kwargs={"scaler": 0.0002}):
        self._scale_time(**kwargs)


# -- TorchNN: ----------------------------------------------------------------------------
class TorchNNIntegrator(BaseForwardIntegrator):
    def __init__(self):
        pass
    
    def forward(self, func, X0, t, dt=0.1, stdev=0.5, max_steps=None):
        diffuser = BrownianDiffuser(X0=X0, t=t, dt=dt, stdev=stdev, max_steps=max_steps)
        return diffuser(func)
