
from . import mix_ins, base


from neural_diffeqs import PotentialODE
import torch_nets
import torch


from typing import Optional, Union, List
from ... import __version__


class LightningODE_VAE_FixedPotential(
    mix_ins.BaseForwardMixIn,
    base.BaseLightningDiffEq,
    mix_ins.PreTrainMixIn,
    mix_ins.PotentialMixIn,
):
    def __init__(
        self,
        data_dim,
        latent_dim: int = 50,
        name: Optional[str] = None,
        mu_hidden: Union[List[int], int] = [2000, 2000],
        mu_activation: Union[str, List[str]] = 'LeakyReLU',
        mu_dropout: Union[float, List[float]] = 0.2,
        mu_bias: bool = True,
        mu_output_bias: bool = True,
        mu_n_augment: int = 0,
        train_lr=1e-5,
        pretrain_lr=1e-3,
        pretrain_epochs=100,
        pretrain_optimizer=torch.optim.Adam,
        train_optimizer=torch.optim.RMSprop,
        pretrain_scheduler=None,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        pretrain_step_size=None,
        train_step_size=10,
        dt=0.1,
        adjoint=False,
        backend = "auto",
        version = __version__,
        *args,
        **kwargs,
    ):
        super().__init__()
        
        name = self._configure_name(name)

        self.save_hyperparameters()

        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(func=PotentialODE, kwargs=locals())
        self._configure_lightning_model(kwargs = locals())

    def forward(self, X0, t, **kwargs):
        """Forward step: (0) integrate in latent space"""
        Z0 = self.Encoder(X0)
        Z_hat = self.integrate(Z0=Z0, t=t, dt=self.hparams["dt"], logqp=False, **kwargs)
        return self.Decoder(Z_hat)

    def step(self, batch, batch_idx, stage=None):

        batch = self.process_batch(batch, batch_idx)
        X_hat = self.forward(batch.X0, batch.t)
        sinkhorn_loss = self.compute_sinkhorn_divergence(
            batch.X, X_hat, batch.W, batch.W_hat
        )
        return self.log_sinkhorn_divergence(sinkhorn_loss).sum()

    def pretrain_step(self, batch, batch_idx, stage=None):

        batch = self.process_batch(batch, batch_idx)
        X0_hat = self.Decoder(self.Encoder(batch.X0))
        recon_loss = self.reconstruction_loss(X0_hat, batch.X0).sum()
        self.log("pretrain_rl_mse", recon_loss.item())
        return recon_loss

    def __repr__(self):
        return "LightningODE-VAE-FixedPotential"