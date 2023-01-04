
__module_name__ = "__init__.py"
__version__ = "0.0.45"
__doc__ = """TODO"""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# -- import packages: ------------------------------------------------------------------
from abc import ABC
from typing import Union, Generator
import numpy as np
import torch


# -- import local dependencies: --------------------------------------------------------
from ._extract_func_kwargs import extract_func_kwargs
from ..loss import Loss
from ..forward import general_forward # SDE_forward
from ..lightning_models import LightningDiffEq, default_NeuralSDE


# -- import packages: ------------------------------------------------------------------
class FunctionFetch(ABC):
    def __parse__(self, kwargs, ignore=["self"]):
        for k, v in kwargs.items():
            if not k in ignore:
                setattr(self, k, v)

    def __init__(self, module=None, parent=None):
        self.__parse__(locals())

    def __call__(self, func):
        if isinstance(func, str):
            return getattr(self.module, func)
        elif issubclass(func, self.parent):
            return func
        else:
            print("pass something else....")

def fetch_optimizer(func):
    """
    Examples:
    ---------
    fetch_optimizer("RMSprop")
    >>> torch.optim.rmsprop.RMSprop

    fetch_optimizer(torch.optim.RMSprop)
    >>> torch.optim.rmsprop.RMSprop
    """
    fetch = FunctionFetch(module=torch.optim, parent=torch.optim.Optimizer)
    return fetch(func)


def fetch_lr_scheduler(func):
    """
    Examples:
    ---------
    fetch_lr_scheduler("StepLR")
    >>> torch.optim.lr_scheduler.StepLR

    fetch_lr_scheduler(torch.optim.lr_scheduler.StepLR)
    >>> torch.optim.lr_scheduler.StepLR
    """
    fetch = FunctionFetch(
        module=torch.optim.lr_scheduler, parent=torch.optim.lr_scheduler._LRScheduler
    )
    return fetch(func)


# -- Main module class: ----------------------------------------------------------------
class LightningModelConfig:
    """
    Called from within the LightningModule. Handles setup of optimizer, lr_scheduler,
    loss function(s), and the forward function.

    TODO:
    -----
    (1) add documentation
    (2) add support for multiple optimizers & schedulers (needed or not?)
        - currently, I have added a counter for each optimizer and scheduler,
          but I would likely have to add a means of using that count to parse
          the passed args (if is list) for each keyword argument.
    """

    def __parse__(self, kwargs, ignore=["self", "func"]):
        for k, v in kwargs.items():
            if not k in ignore:
                setattr(self, "_{}".format(k), v)

    def __init__(
        self,
        func=None,
        adata=None,
        use_key=None,
        time_key="Time point",
        optimizer="RMSprop",
        lr_scheduler="StepLR",
        lr=1e-4,
        step_size=20,
        dt=0.1,
        t=None,
        *args,
        **kwargs,
    ):
        self.__parse__(locals())
        self._configure_func(func)
        self._format_model_params(self._params)
        self._configure_model(self.func)

    def _format_model_params(self, params):
        if isinstance(params, list) and isinstance(
            params[0], torch.nn.parameter.Parameter
        ):
            print("TRUE!")
            self._params = params

        elif isinstance(params, Generator):
            self._params = list(params)

        elif isinstance(params(), Generator):
            self._params = list(params())

        else:
            print("Passed params not being processed properly...")

    @property
    def _optimizer_kwargs(self):
        return extract_func_kwargs(self._fetch_optimizer(self._optimizer), self._kwargs)

    def _fetch_optimizer(self, optimizer: Union[torch.optim.Optimizer, str]):
        return fetch_optimizer(optimizer)

    def _configure_optimizer(self, optimizer: Union[torch.optim.Optimizer, str]):
        self.config_optimizer = self._fetch_optimizer(optimizer)(
            self._params, lr=self._lr, **self._optimizer_kwargs
        )

    @property
    def optimizer(self):
        if not hasattr(self, "config_optimizer"):
            self._configure_optimizer(self._optimizer)
        return self.config_optimizer

    def _fetch_lr_scheduler(
        self, lr_scheduler: Union[torch.optim.lr_scheduler._LRScheduler, str]
    ):
        return fetch_lr_scheduler(lr_scheduler)

    @property
    def _lr_scheduler_kwargs(self):
        return extract_func_kwargs(
            self._fetch_lr_scheduler(self._lr_scheduler), self._kwargs
        )

    def _configure_func(self, func):

        if not func:
            self._kwargs["state_size"] = self._adata.obsm[self._use_key].shape[1]
            neural_sde_kwargs = extract_func_kwargs(default_NeuralSDE, self._kwargs)
            func = default_NeuralSDE(**neural_sde_kwargs)
            
        self.func = func
        self._params = self.func.parameters()
        
    def _configure_lr_scheduler(
        self, lr_scheduler: Union[torch.optim.lr_scheduler._LRScheduler, str]
    ):
        return self._fetch_lr_scheduler(lr_scheduler)(
            optimizer=self.optimizer,
            step_size=self._step_size,
            **self._lr_scheduler_kwargs,
        )
    
    def _configure_forward(self, func):
        pass

    @property
    def lr_scheduler(self):
        return self._configure_lr_scheduler(self._lr_scheduler)

    @property
    def forward_method(self):
        # TODO: flexibility towards NeuralODE and TorchNN
        return general_forward # SDE_forward

    @property
    def loss_function(self):
        return Loss()

    @property
    def dt(self):
        return self._dt
    
    @property
    def t(self):
        if not self._t:
            self._t = torch.Tensor(np.sort(self._adata.obs[self._time_key].unique()))
        return self._t

    def _configure_model(self, func):

        self._LIGHTNING_MODEL = LightningDiffEq(func)
        self._LIGHTNING_MODEL.optimizer = self.optimizer
        self._LIGHTNING_MODEL.lr_scheduler = self.lr_scheduler
        self._LIGHTNING_MODEL.forward = self.forward_method
        self._LIGHTNING_MODEL.loss_func = self.loss_function
        self._LIGHTNING_MODEL.t = self.t
        self._LIGHTNING_MODEL.dt = self.dt

    @property
    def LightningModel(self):
        return self._LIGHTNING_MODEL
