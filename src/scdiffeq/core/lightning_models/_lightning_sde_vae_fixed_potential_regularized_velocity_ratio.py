"""Fixed-potential latent SDE with velocity-ratio regularization."""

import neural_diffeqs
import torch
from typing import List, Optional, Union

from . import base, mix_ins


class LightningSDE_VAE_FixedPotential_RegularizedVelocityRatio(
    mix_ins.PotentialMixIn,
    mix_ins.PreTrainMixIn,
    mix_ins.VAEMixIn,
    mix_ins.RegularizedVelocityRatioMixIn,
    base.BaseLightningDiffEq,
):
    """Match decoded distributions while regularizing drift/diffusion in latent space."""
    def __init__(
        self,
        data_dim,
        latent_dim: int = 50,
        velocity_ratio_params: Optional[dict] = None,
        name: Optional[str] = None,
        mu_hidden: Union[List[int], int] = [400, 400],
        mu_activation: Union[str, List[str]] = "LeakyReLU",
        mu_dropout: Union[float, List[float]] = 0.2,
        mu_bias: bool = True,
        mu_output_bias: bool = True,
        mu_n_augment: int = 0,
        sigma_hidden: Union[List[int], int] = [400, 400],
        sigma_activation: Union[str, List[str]] = "LeakyReLU",
        sigma_dropout: Union[float, List[float]] = 0.2,
        sigma_bias: List[bool] = True,
        sigma_output_bias: bool = True,
        sigma_n_augment: int = 0,
        sde_type="ito",
        noise_type="general",
        brownian_dim=1,
        coef_drift: float = 1.0,
        coef_diffusion: float = 1.0,
        backend="auto",
        train_lr=1e-5,
        train_optimizer=torch.optim.RMSprop,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        train_step_size=10,
        pretrain_lr=1e-3,
        pretrain_epochs=100,
        pretrain_optimizer=torch.optim.Adam,
        pretrain_scheduler=torch.optim.lr_scheduler.StepLR,
        pretrain_step_size=200,
        # -- encoder parameters: ----------------------------------------------
        encoder_n_hidden: int = 4,
        encoder_power: float = 2,
        encoder_activation: Union[str, List[str]] = "LeakyReLU",
        encoder_dropout: Union[float, List[float]] = 0.2,
        encoder_bias: bool = True,
        encoder_output_bias: bool = True,
        # -- decoder parameters: ----------------------------------------------
        decoder_n_hidden: int = 4,
        decoder_power: float = 2,
        decoder_activation: Union[str, List[str]] = "LeakyReLU",
        decoder_dropout: Union[float, List[float]] = 0.2,
        decoder_bias: bool = True,
        decoder_output_bias: bool = True,
        dt: float = 0.1,
        adjoint: bool = False,
        loading_existing: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        if velocity_ratio_params is None:
            velocity_ratio_params = {"target": 2.5, "enforce": 100, "method": "square"}
        else:
            velocity_ratio_params = dict(velocity_ratio_params)
        name = self._configure_name(name, loading_existing=loading_existing)
        self.save_hyperparameters()
        self._configure_torch_modules(func=neural_diffeqs.PotentialSDE, kwargs=locals())
        self._configure_lightning_model(kwargs=locals())

    def pretrain_step(self, batch, batch_idx, stage=None):
        return mix_ins.VAEMixIn.pretrain_step(self, batch, batch_idx, stage=stage)

    def step(self, batch, batch_idx, stage=None):
        if stage == "training":
            optimizer = self.optimizers()[1]
            optimizer.zero_grad()

        batch = self.process_batch(batch, batch_idx)
        latent = self.integrate(
            Z0=self.encode(batch.X0), t=batch.t, dt=self.hparams["dt"], logqp=False,
        )
        self.sinkhorn_loss = self.compute_sinkhorn_divergence(
            batch.X, self.decode(latent), batch.W, batch.W_hat,
        )
        self.velocity_ratio_loss = self._velocity_ratio_loss(latent)
        self.total_loss = self.sinkhorn_loss + self.velocity_ratio_loss
        loss = self.total_loss.sum()
        if stage == "training":
            self.manual_backward(loss)
            self._clip_vae_gradients(optimizer)
            optimizer.step()
            self.lr_schedulers()[1].step()
        return loss.detach()

    def __repr__(self):
        return "LightningSDE-VAE-FixedPotential-RegularizedVelocityRatio"
