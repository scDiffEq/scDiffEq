

# -- import packages: --------------------------------------------------------------------
from autodevice import AutoDevice
import numpy as np
import torch


# -- import local dependencies: ----------------------------------------------------------
from ...utils import AutoParseBase, extract_func_kwargs
from ._function_credentials import Credentials


# -- Main module class: ------------------------------------------------------------------
class UniversalForwardIntegrator(AutoParseBase):
    def __init__(self, func, adjoint=False):

        self.func = func
        creds = Credentials(self.func, adjoint=adjoint)
        self.func_type, self.mu_is_potential, self.sigma_is_potential = creds()
        self.integrator = creds.integrator

    def __call__(
        self,
        X0,
        t,
        dt=0.1,
        stdev=torch.Tensor([0.5]),
        device=AutoDevice(),
        max_steps=None,
        fate_scale=0,
        return_all=False,
    ):
        """
        Notes:
        ------
        (1) For some reason, locals(), which gets passed on to  self._KWARGS as part of  the
            Base class, is carrying all arguments from the main model class. For now, I need
            to manually remove / pop these.
        """
        self.__parse__(locals(), ignore=["self", "X0", "func", "t"])
        int_kwargs = extract_func_kwargs(func=self.integrator, kwargs=self._KWARGS)

        if "func" in int_kwargs.keys():
            func = int_kwargs.pop("func")
        if "t" in int_kwargs.keys():
            _t = int_kwargs.pop("t").to(device)
        return self.integrator(self.func, X0, t, **int_kwargs)
