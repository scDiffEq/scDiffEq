
__module_name__ = "_ForwardIntegrators.py"
__doc__ = """Base module for classes that handle forward integration."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])

# -- import packages: --------------------------------------------------------------------
import torch
import torchsde
import torchdiffeq

# -- import local dependencies: ----------------------------------------------------------
from ._BaseForwardIntegrators import BaseForwardIntegrator


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