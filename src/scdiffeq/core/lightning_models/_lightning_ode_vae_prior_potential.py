# -- import packages: ---------------------------------------------------------
from neural_diffeqs import LatentPotentialODE
import torch


# -- import local dependencies: -----------------------------------------------
from . import mix_ins
from . import base


# -- set type hints: ----------------------------------------------------------
from typing import Optional, Union, List


# -- lightning model: ---------------------------------------------------------
class LightningODE_VAE_PriorPotential(
    mix_ins.DriftPriorMixIn,
    mix_ins.PotentialMixIn,
    mix_ins.VAEMixIn,
    mix_ins.PreTrainMixIn,
    base.BaseLightningDiffEq,
):
    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 50,
        name: Optional[str] = None,
        train_lr: float = 1e-5,
        pretrain_lr: float = 1e-3,
        pretrain_epochs: int = 100,
        pretrain_optimizer: torch.optim.Optimizer = torch.optim.Adam,
        train_optimizer: torch.optim.Optimizer = torch.optim.RMSprop,
        pretrain_scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.StepLR,
        train_scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.StepLR,
        pretrain_step_size: int = 200,
        train_step_size: int = 10,
        dt: float = 0.1,
        adjoint: bool = False,
        backend: str = "auto",
        # -- encoder parameters: -------
        encoder_n_hidden: int = 4,
        encoder_power: float = 2,
        encoder_activation: Union[str, List[str]] = "LeakyReLU",
        encoder_dropout: Union[float, List[float]] = 0.2,
        encoder_bias: bool = True,
        encoder_output_bias: bool = True,
        # -- decoder parameters: -------
        decoder_n_hidden: int = 4,
        decoder_power: float = 2,
        decoder_activation: Union[str, List[str]] = "LeakyReLU",
        decoder_dropout: Union[float, List[float]] = 0.2,
        decoder_bias: bool = True,
        decoder_output_bias: bool = True,
        loading_existing: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        LightningODE_VAE_PriorPotential

        Parameters:
        -----------
        data_dim : int
            Dimensionality of the input data
        latent_dim : int, optional
            Dimensionality of the latent space, by default 50
        name : str, optional
            Name of the model, by default None
        train_lr : float, optional
            Learning rate for training, by default 1e-5
        pretrain_lr : float, optional
            Learning rate for pretraining, by default 1e-3
        pretrain_epochs : int, optional
            Number of epochs for pretraining, by default 100
        pretrain_optimizer : torch.optim.Optimizer, optional
            Optimizer for pretraining, by default torch.optim.Adam
        train_optimizer : torch.optim.Optimizer, optional
            Optimizer for training, by default torch.optim.RMSprop
        pretrain_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler for pretraining, by default torch.optim.lr_scheduler.StepLR
        train_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler for training, by default torch.optim.lr_scheduler.StepLR
        pretrain_step_size : int, optional
            Step size for the pretraining learning rate scheduler, by default 200
        train_step_size : int, optional
            Step size for the training learning rate scheduler, by default 10
        dt : float, optional
            Time step for the ODE solver, by default 0.1
        adjoint : bool, optional
            Whether to use the adjoint method for the ODE solver, by default False
        backend : str, optional
            Backend for the ODE solver, by default "auto"
        encoder_n_hidden : int, optional
            Number of hidden layers in the encoder, by default 4
        encoder_power : float, optional
            Power of the encoder, by default 2
        encoder_activation : Union[str, List[str]], optional
            Activation function(s) for the encoder, by default 'LeakyReLU'
        encoder_dropout : Union[float, List[float]], optional
            Dropout rate(s) for the encoder, by default 0.2
        encoder_bias : bool, optional
            Whether to use bias in the encoder, by default True
        encoder_output_bias : bool, optional
            Whether to use bias in the output layer of the encoder, by default True
        decoder_n_hidden : int, optional
            Number of hidden layers in the decoder, by default 4
        decoder_power : float, optional
            Power of the decoder, by default 2
        decoder_activation : Union[str, List[str]], optional
            Activation function(s) for the decoder, by default 'LeakyReLU'
        decoder_dropout : Union[float, List[float]], optional
            Dropout rate(s) for the decoder, by default 0.2
        decoder_bias : bool, optional
            Whether to use bias in the decoder, by default True
        decoder_output_bias : bool, optional
            Whether to use bias in the output layer of the decoder, by default True
        loading_existing : bool, optional
            Whether to load an existing model, by default False

        Returns:
        --------
        None

        Notes:
        ------
        This class implements a VAE with prior potential ODE using PyTorch Lightning.

        Examples:
        ---------
        >>> model = LightningODE_VAE_PriorPotential(data_dim=100, latent_dim=20, dt=0.05)
        >>> model.fit(data)
        """
        super().__init__()

        name = self._configure_name(name, loading_existing=loading_existing)

        self.save_hyperparameters()

        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(func=LatentPotentialODE, kwargs=locals())
        self._configure_lightning_model(kwargs=locals())

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
