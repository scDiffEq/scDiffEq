
from . import mix_ins, base


from neural_diffeqs import PotentialSDE
import torch_nets
import torch

from typing import Optional, Union, List

from ... import __version__

class LightningSDE_VAE_FixedPotential(
    mix_ins.PreTrainMixIn,
    mix_ins.PotentialMixIn,
    mix_ins.VAEMixIn,
    base.BaseLightningDiffEq,
):
    def __init__(
        self,
        data_dim,
        latent_dim: int = 50,
        name: Optional[str] = None,
        mu_hidden: Union[List[int], int] = [400, 400],
        mu_activation: Union[str, List[str]] = 'LeakyReLU',
        mu_dropout: Union[float, List[float]] = 0.2,
        mu_bias: bool = True,
        mu_output_bias: bool = True,
        mu_n_augment: int = 0,
        sigma_hidden: Union[List[int], int] = [400, 400],
        sigma_activation: Union[str, List[str]] = 'LeakyReLU',
        sigma_dropout: Union[float, List[float]] = 0.2,
        sigma_bias: List[bool] = True,
        sigma_output_bias: bool = True,
        sigma_n_augment: int = 0,
        sde_type='ito',
        noise_type='general',
        brownian_dim=1,
        coef_drift: float = 1.0,
        coef_diffusion: float = 1.0,
        backend = "auto",
        
        train_lr=1e-5,
        train_optimizer=torch.optim.RMSprop,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        train_step_size=10,
        
        pretrain_lr=1e-3,
        pretrain_epochs=100,
        pretrain_optimizer=torch.optim.Adam,
        pretrain_scheduler=torch.optim.lr_scheduler.StepLR,
        pretrain_step_size=200,
        
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
        
        dt=0.1,
        adjoint=False,
        version = __version__,
        
        *args,
        **kwargs,
    ):
        super().__init__()
        
        name = self._configure_name(name)

        self.save_hyperparameters()
        
        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(func = PotentialSDE, kwargs=locals())
        self._configure_lightning_model(kwargs = locals())

    def pretrain_step(self, batch, batch_idx, stage=None):

        batch = self.process_batch(batch, batch_idx)
        X0_hat = self.Decoder(self.Encoder(batch.X0))
        recon_loss = self.reconstruction_loss(X0_hat, batch.X0).sum()
        self.log("pretrain_rl_mse", recon_loss.item())
        return recon_loss

    def training_step(self, batch, batch_idx, *args, **kwargs):
        if self.PRETRAIN:
            return self.pretrain_step(batch, batch_idx, stage="pretrain")
        return self.step(batch, batch_idx, stage="training")

    def __repr__(self):
        return "LightningSDE-VAE-FixedPotential"
