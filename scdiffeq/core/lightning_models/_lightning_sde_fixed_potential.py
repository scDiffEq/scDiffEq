
from neural_diffeqs import PotentialSDE
import torch


from . import base, mix_ins
from typing import Union, List

class LightningSDE_FixedPotential(
    base.BaseLightningDiffEq,
    mix_ins.PotentialMixIn,
    mix_ins.BaseForwardMixIn,
):
    def __init__(
        self,
        latent_dim,
        mu_hidden: Union[List[int], int] = [400, 400, 400],
        sigma_hidden: Union[List[int], int] = [400, 400, 400],
        mu_activation: Union[str, List[str]] = 'LeakyReLU',
        sigma_activation: Union[str, List[str]] = 'LeakyReLU',
        mu_dropout: Union[float, List[float]] = 0.1,
        sigma_dropout: Union[float, List[float]] = 0.1,
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
        
        train_lr=1e-4,
        train_optimizer=torch.optim.RMSprop,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        train_step_size=10,
        dt=0.1,
        adjoint=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        
        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(func=PotentialSDE, kwargs=locals())
        self._configure_optimizers_schedulers()

    def __repr__(self):
        return "LightningSDE-FixedPotential"
