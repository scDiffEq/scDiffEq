
from neural_diffeqs import LatentPotentialODE
import torch

from . import base, mix_ins

from .. import utils

from typing import Optional, Union, List
from ... import __version__


class LightningODE_PriorPotential(
    base.BaseLightningDiffEq,
    
):
    def __init__(
        self,
        # -- general params: ---------------------------------------------------
        latent_dim: int = 50,
        name: Optional[str] = None,
        train_lr=1e-5,
        train_optimizer=torch.optim.RMSprop,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        train_step_size=10,
        adjoint=False,
        
        # -- ODE params: -------------------------------------------------------
        coef_diff: float = 0,
        dt: float = 0.1,
        mu_hidden: Union[List[int], int] = [400, 400],
        mu_activation: Union[str, List[str]] = 'LeakyReLU',
        mu_dropout: Union[float, List[float]] = 0.2,
        mu_bias: bool = True,
        mu_output_bias: bool = True,
        mu_n_augment: int = 0,
        sde_type='ito',
        noise_type='general',
        backend = "auto",
        brownian_dim=1,
        
        version = __version__,
        
        *args,
        **kwargs,
    ):
        super().__init__()
        
        name = self._configure_name(name)

        self.save_hyperparameters()        
        self.func = LatentPotentialODE(
            state_size=latent_dim,
            **utils.extract_func_kwargs(func=LatentPotentialODE, kwargs=locals()),
        )
        self._configure_lightning_model(kwargs = locals())
        
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
        return "LightningODE-PriorPotential"