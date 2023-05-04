
# -- import packages: ------
import brownian_diffuser
import neural_diffeqs
import torchdiffeq
import torchsde
import torch


# -- import local dependencies: ------
from .. import lightning_models, utils


# -- supporting functions: ------------------------------------
def fetch_optimizer(func):
    """
    Examples:
    ---------
    fetch_optimizer("RMSprop")
    >>> torch.optim.rmsprop.RMSprop

    fetch_optimizer(torch.optim.RMSprop)
    >>> torch.optim.rmsprop.RMSprop
    """
    fetch = utils.FunctionFetch(module=torch.optim, parent=torch.optim.Optimizer)
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
    fetch = utils.FunctionFetch(
        module=torch.optim.lr_scheduler, parent=torch.optim.lr_scheduler._LRScheduler
    )
    return fetch(func)


# class ConfigureLightningModel:
class ChooseLitModel:
    def __init__(self, data_dim, latent_dim, de_type="SDE", potential_type=None):
        """
        data_dim
            [ required ]

        latent_dim
            [ required ]

        de_type
            options: ["SDE", "ODE"]
            default: "ODE"

        potential_type
            options: [None, "fixed", "prior"]
            default: None
        """

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self._de_type = de_type
        self._potential_type = potential_type

    @property
    def de_type(self):
        return "".join(["Lightning", self._de_type])

    @property
    def potential_type(self):
        potential_types = {
            "fixed": "FixedPotential",
            "prior": "PriorPotential",
        }
        if self._potential_type:
            return potential_types[self._potential_type]

    @property
    def use_vae(self):
        return self.data_dim > self.latent_dim

    @property
    def model(self):

        _model = [self.de_type]

        if self.use_vae:
            _model.append("VAE")

        if self.potential_type:
            _model.append(self.potential_type)

        return "_".join(_model)


# -- main operator class: ------------------------------------------------------------
class LightningModelConfiguration:
    def __init__(
        self,
        data_dim,
        latent_dim,
        de_type="SDE",
        potential_type=None,
#         func,
        optimizer=torch.optim.RMSprop,
        lr_scheduler=torch.optim.lr_scheduler.StepLR,
        adjoint=False,
    ):
        
        self._lit_model_choice = ChooseLitModel(
            data_dim=data_dim,
            latent_dim=latent_dim,
            de_type=de_type,
            potential_type=potential_type,
        )
#         self.func = func
        self._adjoint = adjoint
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler

#         self.func_type, self.mu_is_potential, self.sigma_is_potential = self.function_credentials

#     @property
#     def function_credentials(self):
#         if isinstance(self.func, neural_diffeqs.NeuralODE):
#             if self._adjoint:
#                 integrator = torchdiffeq.odeint_adjoint
#             else:
#                 integrator = torchdiffeq.odeint
#             return "NeuralODE", self._is_mu_potential(), self._is_sigma_potential()

#         if isinstance(self.func, neural_diffeqs.NeuralSDE):
#             if self._adjoint:
#                 integrator = torchsde.sdeint_adjoint
#             else:
#                 integrator = torchsde.sdeint
#             return "NeuralSDE", self._is_mu_potential(), self._is_sigma_potential()

#         integrator = brownian_diffuser.nn_int
#         self._integrator = brownian_diffuser.nn_int
#         return "DriftNet", self._is_mu_potential(), self._is_sigma_potential()
    
#     def _is_mu_potential(self):
#         """Assumes potential is 1-D"""
# #         if not isinstance(self.func, neural_diffeqs.NeuralODE):
# #             return False
#         return list(self.func.mu.parameters())[-1].shape[0] == 1

#     def _is_sigma_potential(self):
#         """Assumes potential is 1-D"""
#         if not isinstance(self.func, neural_diffeqs.NeuralSDE):
#             return False
#         return list(self.func.sigma.parameters())[-1].shape[0] == 1

    def _non_import(self, attr):
        return (
            attr.startswith("_")
            or attr.startswith("Base")
            or (attr in ["SinkhornDivergence"])
        )

    @property
    def available_models(self):
        """Available, implemented models"""
        return {
            attr: getattr(lightning_models, attr)
            for attr in lightning_models.__dir__()
            if not self._non_import(attr)
        }

    @property
    def FunctionModuleName(self):
        func_type = self.func_type.strip("Neural")
        if self.mu_is_potential:
            return "".join(["LightningPotential"] + [func_type])
        return "".join(["Lightning"] + [func_type])
    
    @property
    def optimizer(self):
        return fetch_optimizer(self._optimizer)
    
    @property
    def lr_scheduler(self):
        return fetch_lr_scheduler(self._lr_scheduler)


    def __call__(self, kwargs):
        """Return the LightningModel"""
        model = self.available_models[self.FunctionModuleName]
        MODEL_KWARGS = utils.extract_func_kwargs(func = model, kwargs = kwargs, ignore=['optimizer', 'lr_scheduler'])
        return model(
            func=self.func,
            optimizer = self.optimizer,
            lr_scheduler = self.lr_scheduler,
            **MODEL_KWARGS,
        )
