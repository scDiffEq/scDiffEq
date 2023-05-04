
from neural_diffeqs import LatentPotentialSDE
import torch

from . import base, mix_ins

from .. import utils

from typing import Union, List

class LightningSDE_PriorPotential(
    base.BaseLightningDiffEq,
    mix_ins.PotentialMixIn,
):
    def __init__(
        self,
        latent_dim,
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
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        
        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(func=LatentPotentialSDE, kwargs=locals())
        self._configure_optimizers_schedulers()

    def forward(self, X0, t, **kwargs):
        """Forward step: (0) integrate in latent space"""
        Z_hat, KL_div = self.integrate(
            Z0=X0, t=t, dt=self.hparams["dt"], logqp=True, **kwargs
        )
        return Z_hat, KL_div

    def log_computed_loss(self, sinkhorn_loss, t, kl_div_loss, stage):
        
        sinkhorn_loss = self.log_sinkhorn_divergence(sinkhorn_loss).sum()
        self.log(f"kl_div_{stage}", kl_div_loss.sum())

        return sinkhorn_loss + kl_div_loss.sum()

    def step(self, batch, batch_idx, stage=None):

        batch = self.process_batch(batch, batch_idx)
        X_hat, kl_div_loss = self.forward(batch.X0, batch.t)
        sinkhorn_loss = self.compute_sinkhorn_divergence(
            batch.X, X_hat, batch.W, batch.W_hat
        )
        return self.log_computed_loss(
            sinkhorn_loss, t=batch.t, kl_div_loss=kl_div_loss, stage=stage
        )

        
    def __repr__(self):
        return "LightningSDE-PriorPotential"