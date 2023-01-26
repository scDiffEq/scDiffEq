
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
from ..lightning_model import LightningDiffEq, default_NeuralSDE
from ..lightning_model.forward import Credentials
from ..utils import extract_func_kwargs, AutoParseBase



from autodevice import AutoDevice

NoneType = type(None)


# -- import packages: ------------------------------------------------------------------
class FunctionFetch(AutoParseBase):
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

def kwargs_from_private(kwargs):
    unprivate = {}
    for k in kwargs.keys():
        unprivate[k[1:]] = kwargs[k]
        
    return unprivate


# -- Main module class: ----------------------------------------------------------------
class LightningModelConfig(AutoParseBase):
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

    def __init__(
        self,
        func=None,
        adata=None,
        seed=0,
        use_key=None,
        time_key="Time point",
        optimizers=["SGD","RMSprop",],
        lr_schedulers=["StepLR", "StepLR"],
        learning_rates = [1e-8, 1e-4],
        stdev=torch.nn.Parameter(torch.tensor(0.5, requires_grad=True, device=AutoDevice())),
        V_scaling=torch.nn.Parameter(torch.tensor(1.0, requires_grad=True, device=AutoDevice())),
        V_coefficient=0.2,
        t0_idx = None,
        adjoint=False,
        fate_scale=10,
        step_size=20,
        dt=0.1,
        t=None,
        n_steps=40,
        tau=1e-06,
        burn_steps=100,
        disable_velocity=False,
        disable_potential=False,
        disable_fate_bias=False,
        skip_positional_backprop=False,
        skip_positional_velocity_backprop=False,
        skip_potential_backprop=False,
        skip_fate_bias_backprop=False,
        velo_gene_idx = None, # self._adata.uns["velo_gene_idx"]
        *args,
        **kwargs,
    ):
        super(LightningModelConfig, self).__init__()
                
        self.__parse__(locals(), ignore=["self", "func", "stdev"], public=[None])
                
        if not isinstance(self._optimizers, list):
            self._optimizers = [self._optimizers]
        if not isinstance(self._lr_schedulers, list):
            self._lr_schedulers = [self._lr_schedulers]

        n_sch = len(self._lr_schedulers)
        n_opt = len(self._optimizers)
        if n_sch < n_opt:
            n_diff = n_opt - n_sch
            self._lr_schedulers = self._lr_schedulers + [self._lr_schedulers[-1]]+n_diff
        
        self.stdev = stdev
        self.V_scaling = V_scaling
        self.V_coefficient = V_coefficient
        
        self._configure_func(func)
        self._format_model_params(self._params)
        self._configure_model(self.func, adjoint)
        
        

    def _format_model_params(self, params):
        if isinstance(params, list) and isinstance(
            params[0], torch.nn.parameter.Parameter
        ):
            self._params = params

        elif isinstance(params, Generator):
            self._params = list(params)

        elif isinstance(params(), Generator):
            self._params = list(params())

        else:
            print("Passed params not being processed properly...")

    @property
    def _optimizer_kwargs(self):
        _kw = {}
        for optim in self._optimizers:
            _kw.update(extract_func_kwargs(fetch_optimizer(optim), kwargs_from_private(self._KWARGS)))
        return _kw

        
    @property
    def is_potential_net(self, net):
        """Assumes potential is 1-D"""
        return list(net.parameters())[-1].shape[0] == 1

    @property
    def optimizers(self):
        if not hasattr(self, "_configured_optimizers"):
            
            

            self._configured_optimizers = [
                fetch_optimizer(optim)(self._params, lr=self._learning_rates[n],
                                       **self._optimizer_kwargs)
                for n, optim in enumerate(self._optimizers)
            ]
        return self._configured_optimizers

#     def _fetch_lr_scheduler(
#         self, lr_scheduler: Union[torch.optim.lr_scheduler._LRScheduler, str]
#     ):
#         return fetch_lr_scheduler(LRS) for LRS in self._lr_schedulers

    @property
    def _lr_scheduler_kwargs(self):
        _kw = {}
        for LRS in self._lr_schedulers:
            _kw.update(extract_func_kwargs(fetch_lr_scheduler(LRS), kwargs_from_private(self._KWARGS)))
        return _kw
#         return extract_func_kwargs(
#             self._fetch_lr_scheduler(self._lr_scheduler), self._KWARGS
#         )

    @property
    def real_time(self):
        return isinstance(self._t0_idx, NoneType)
        
    def _configure_func(self, func):

        if not func:
            self._KWARGS["state_size"] = self._adata.obsm[self._use_key].shape[1]
            neural_sde_kwargs = extract_func_kwargs(default_NeuralSDE, self._KWARGS)
            func = default_NeuralSDE(**neural_sde_kwargs)
        
        self.func = func
        _params = list(self.func.parameters())
        
        for p in [self.stdev, self.V_scaling]:
            if isinstance(p, torch.nn.parameter.Parameter):
                _params = _params + [p]
        self._params = _params

    def _configure_lr_scheduler(
        self, lr_scheduler: Union[torch.optim.lr_scheduler._LRScheduler, str]
    ):
        return self._fetch_lr_scheduler(lr_scheduler)(
            optimizer=self.optimizer,
            step_size=self._step_size,
            **self._lr_scheduler_kwargs,
        )
    
    @property
    def forward(self):
        return forward

    @property
    def lr_schedulers(self):
        schedulers = [fetch_lr_scheduler(LRS) for LRS in self._lr_schedulers]
        return [scheduler(optim, **self._lr_scheduler_kwargs) for optim, scheduler in zip(self._configured_optimizers, schedulers)]
#         return self._configure_lr_scheduler()

    @property
    def loss_function(self):
        return Loss()

    @property
    def dt(self):
        return self._dt
    
    @property
    def t(self):
        return self._t
    
    @property
    def base_ignore(self):
        return ['self', '__class__', 'lightning_ignore', 'base_ignore']
    
    @property
    def lightning_ignore(self):
        return [
            'self',
            '__class__',
            'func',
            'adata',
            'stdev',
            'V_scaling',
            'ignore',
            't',
            'optimizer',
            'lr_scheduler',
            'lightning_ignore',
            'base_ignore',
               ]
    
    # -- function credentialling: --------------------------------------------------------
    def _configure_function_credentials(self, func, adjoint):
        
        creds = Credentials(self.func, adjoint)
        self.func_type, self.mu_is_potential, self.sigma_is_potential = creds()
        
    def _configure_model(self, func, adjoint):
        
        self._configure_function_credentials(func, adjoint)
        
        self._LIGHTNING_MODEL = LightningDiffEq(
            adata=self._adata,
            func=func,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            stdev=self.stdev,
            expand=False,
            t=self.t,
            dt=self.dt,
            use_key=self._use_key,
            time_key=self._time_key,
            adjoint = self._adjoint,
            real_time = self.real_time,
            burn_steps = self._burn_steps,
            tau=self._tau,
            velo_gene_idx=self._velo_gene_idx,
            fate_scale=self._fate_scale,
            V_coefficient = self._V_coefficient,
            V_scaling = self._V_scaling,
            seed = self._seed,
            func_type=self.func_type,
            mu_is_potential=self.mu_is_potential,
            sigma_is_potential=self.sigma_is_potential,
            lightning_ignore=self.lightning_ignore,
            base_ignore=self.base_ignore,
        )

    @property
    def LightningModel(self):
        return self._LIGHTNING_MODEL

