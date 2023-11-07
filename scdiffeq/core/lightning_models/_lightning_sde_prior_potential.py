
from neural_diffeqs import LatentPotentialSDE
import torch

from . import base, mix_ins

from .. import utils

from typing import Optional, Union, List

from ... import __version__

class LightningSDE_PriorPotential(
    mix_ins.PotentialMixIn,
    mix_ins.DriftPriorMixIn,
    base.BaseLightningDiffEq,
):
    def __init__(
        self,
        latent_dim: int = 50,
        name: Optional[str] = None,
        train_lr=1e-4,
        train_optimizer=torch.optim.RMSprop,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        train_step_size=10,
        dt=0.1,
        adjoint=False,
        mu_hidden: Union[List[int], int] = [400, 400, 400],
        sigma_hidden: Union[List[int], int] = [400, 400, 400],
        mu_activation: Union[str, List[str]] = 'LeakyReLU',
        sigma_activation: Union[str, List[str]] = 'LeakyReLU',
        mu_dropout: Union[float, List[float]] = 0.2,
        sigma_dropout: Union[float, List[float]] = 0.2,
        mu_bias: bool = True,
        sigma_bias: List[bool] = True,
        mu_output_bias: bool = True,
        sigma_output_bias: bool = True,
        mu_n_augment: int = 0,
        sigma_n_augment: int = 0,
        sde_type='ito',
        noise_type='general',
        brownian_dim=1,
        coef_drift: float = 1.0,
        coef_diffusion: float = 1.0,
        coef_prior_drift: float = 1.0,
        backend = "auto",
        version = __version__,
        *args,
        **kwargs,
    ):
        super().__init__()
        
        name = self._configure_name(name)

        self.save_hyperparameters()
        
        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(func=LatentPotentialSDE, kwargs=locals())
        self._configure_lightning_model(kwargs = locals())
        
    def __repr__(self):
        return "LightningSDE-PriorPotential"