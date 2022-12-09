
__module_name__ = "_base_forward_integrator.py"
__version__ = "0.0.44"
__doc__ = """Base module for classes that handle forward integration."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# -- import packages: --------------------------------------------------------------------
from abc import ABC, abstractmethod
import torch


class BaseForwardIntegrator(ABC):
    """
    Base class for forward integration of a neural differential equation or more
    generally, a neural network passed through a diffuser.
    """
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

    def __call__(
        self,
        func,
        X0,
        t=None,
        ts=None,
        dt=None,
        dt_min=None,
        stdev=None,
        max_steps=None,
        config_optionals=False,
    ):
        """
        handle case-specific conversions then return forward func
        """
        self._config_kwargs(locals())
        if config_optionals:
            self._config_optionals(config_optionals)
            
        return self.forward(func.to("cuda:0"), X0.to("cuda:0"), **self.kwargs)