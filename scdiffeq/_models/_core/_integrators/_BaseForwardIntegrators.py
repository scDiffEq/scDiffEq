
__module_name__ = "_BaseForwardIntegrators.py"
__doc__ = """Base module for classes that handle forward integration."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# -- import packages: --------------------------------------------------------------------
from abc import ABC, abstractmethod
import torch


# -- Integrator base classes: ------------------------------------------------------------
class AbstractForwardIntegrator(ABC):
    @abstractmethod
    def __call__(self):
        pass


class BaseForwardIntegrator(AbstractForwardIntegrator):
    def __init__(self):
        super(BaseForwardIntegrator, self).__init__()

    # -- base integrator supporting functions: -------------------------------------------
    def _specify_forward_module(self, pkg, module):
        setattr(self, "forward", getattr(pkg, module))

    def _config_kwargs(self, local_args):

        """
        parse locally-passed kwargs and return what you
        need for a specific forward integration function

        Ignore self and positional args: func, X0.
        """
        ignore = ["self", "func", "X0", "config_optionals"]
        self.kwargs = {}
        for arg, val in local_args.items():
            if (not arg in ignore) and (isinstance(val, (float, int, torch.Tensor))):
                self.kwargs[arg] = val
    
    def _config_optionals(self, **kwargs):
        """run something INSIDE OF THIS like time_config which can be overwritten on the fly"""
        pass
    
    # -- main call: ----------------------------------------------------------------------
    def __call__(self, func, X0, t=None, ts=None, dt=None, dt_min=None, config_optionals=False):
        """
        handle case-specific conversions then return forward func
        """
        self._config_kwargs(locals())
        if config_optionals:
            self._config_optionals(config_optionals)
        return self.forward(func, X0, **self.kwargs)
