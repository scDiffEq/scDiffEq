
# -- import packages: ----------------------------------------------------------
from neural_diffeqs import LatentPotentialODE
import torch_nets
import torch


# -- import local dependencies: ------------------------------------------------
from . import mix_ins
from . import base


from typing import Optional, Union, List
from ... import __version__


# -- lightning model: ----------------------------------------------------------
class LightningODE_VAE_PriorPotential(
    mix_ins.DriftPriorMixIn,
    mix_ins.PotentialMixIn,
    mix_ins.VAEMixIn,
    mix_ins.PreTrainMixIn,
    base.BaseLightningDiffEq,
):
    def __init__(
        self,
        data_dim,
        latent_dim: int = 50,
        name: Optional[str] = None,
        train_lr=1e-5,
        pretrain_lr=1e-3,
        pretrain_epochs=100,
        pretrain_optimizer=torch.optim.Adam,
        train_optimizer=torch.optim.RMSprop,
        pretrain_scheduler=torch.optim.lr_scheduler.StepLR,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        pretrain_step_size=200,
        train_step_size=10,
        dt=0.1,
        adjoint=False,
        backend = "auto",
        
        # -- encoder parameters: -------
        encoder_n_hidden: int = 4,
        encoder_power: float = 2,
        encoder_activation: Union[str, List[str]] = 'LeakyReLU',
        encoder_dropout: Union[float, List[float]] = 0.2,
        encoder_bias: bool = True,
        encoder_output_bias: bool = True,

        # -- decoder parameters: -------
        decoder_n_hidden: int = 4,
        decoder_power: float = 2,
        decoder_activation: Union[str, List[str]] = 'LeakyReLU',
        decoder_dropout: Union[float, List[float]] = 0.2,
        decoder_bias: bool = True,
        decoder_output_bias: bool = True,
        
        version = __version__,
        
        *args,
        **kwargs,
    ):
        super().__init__()
        
        name = self._configure_name(name)

        self.save_hyperparameters()
        
        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(func = LatentPotentialODE, kwargs=locals())
        self._configure_lightning_model(kwargs = locals())

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

    def pretrain_step(self, batch, batch_idx, stage=None):

        batch = self.process_batch(batch, batch_idx)
        X0_hat = self.Decoder(self.Encoder(batch.X0))
        recon_loss = self.reconstruction_loss(X0_hat, batch.X0).sum()
        self.log("pretrain_rl_mse", recon_loss.item())
        return recon_loss

    def __repr__(self):
        return "LightningODE-VAE-PriorPotential"