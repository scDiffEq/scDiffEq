# -- import packages: ---------------------------------------------------------
import neural_diffeqs
import torch

# -- import local dependencies: -----------------------------------------------
from . import base, mix_ins

# -- set type hints: ----------------------------------------------------------
from typing import Literal, Optional, Union, List


# -- operational class: -------------------------------------------------------
class LightningSDE_VAE_FixedPotential(
    mix_ins.PreTrainMixIn,
    mix_ins.PotentialMixIn,
    mix_ins.VAEMixIn,
    base.BaseLightningDiffEq,
):
    """LightningSDE-VAE-FixedPotential"""
    def __init__(
        self,
        data_dim,
        latent_dim: int = 50,
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
        """
        LightningSDE_VAE_FixedPotential

        Parameters
        ----------
        latent_dim : int, optional
            Dimensionality of the latent space, by default 50.
        name : str, optional
            Name of the model, by default None.
        mu_hidden : Union[List[int], int], optional
            Hidden layer sizes for the drift neural network, by default [400, 400].
        mu_activation : Union[str, List[str]], optional
            Activation function(s) for the drift neural network, by default 'LeakyReLU'.
        mu_dropout : Union[float, List[float]], optional
            Dropout rate(s) for the drift neural network, by default 0.2.
        mu_bias : bool, optional
            Whether to use bias in the drift neural network, by default True.
        mu_output_bias : bool, optional
            Whether to use bias in the output layer of the drift neural network, by default True.
        mu_n_augment : int, optional
            Number of augmentations for the drift neural network, by default 0.
        sigma_hidden : Union[List[int], int], optional
            Hidden layer sizes for the diffusion neural network, by default [400, 400].
        sigma_activation : Union[str, List[str]], optional
            Activation function(s) for the diffusion neural network, by default 'LeakyReLU'.
        sigma_dropout : Union[float, List[float]], optional
            Dropout rate(s) for the diffusion neural network, by default 0.2.
        sigma_bias : List[bool], optional
            Whether to use bias in the diffusion neural network, by default True.
        sigma_output_bias : bool, optional
            Whether to use bias in the output layer of the diffusion neural network, by default True.
        sigma_n_augment : int, optional
            Number of augmentations for the diffusion neural network, by default 0.
        sde_type : str, optional
            Type of stochastic differential equation, by default 'ito'.
        noise_type : str, optional
            Type of noise, by default 'general'.
        brownian_dim : int, optional
            Dimensionality of the Brownian motion, by default 1.
        coef_drift : float, optional
            Coefficient of drift, by default 1.0.
        coef_diffusion : float, optional
            Coefficient of diffusion, by default 1.0.
        backend : str, optional
            Backend for the SDE solver, by default "auto".
        train_lr : float, optional
            Learning rate for training, by default 1e-5.
        train_optimizer : torch.optim.Optimizer, optional
            Optimizer for training, by default torch.optim.RMSprop.
        train_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler for training, by default torch.optim.lr_scheduler.StepLR.
        train_step_size : int, optional
            Step size for the training learning rate scheduler, by default 10.
        pretrain_lr : float, optional
            Learning rate for pretraining, by default 1e-3.
        pretrain_epochs : int, optional
            Number of epochs for pretraining, by default 100.
        pretrain_optimizer : torch.optim.Optimizer, optional
            Optimizer for pretraining, by default torch.optim.Adam.
        pretrain_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler for pretraining, by default torch.optim.lr_scheduler.StepLR.
        pretrain_step_size : int, optional
            Step size for the pretraining learning rate scheduler, by default 200.
        encoder_n_hidden : int, optional
            Number of hidden layers in the encoder, by default 4.
        encoder_power : float, optional
            Power of the encoder, by default 2.
        encoder_activation : Union[str, List[str]], optional
            Activation function(s) for the encoder, by default 'LeakyReLU'.
        encoder_dropout : Union[float, List[float]], optional
            Dropout rate(s) for the encoder, by default 0.2.
        encoder_bias : bool, optional
            Whether to use bias in the encoder, by default True.
        encoder_output_bias : bool, optional
            Whether to use bias in the output layer of the encoder, by default True.
        decoder_n_hidden : int, optional
            Number of hidden layers in the decoder, by default 4.
        decoder_power : float, optional
            Power of the decoder, by default 2.
        decoder_activation : Union[str, List[str]], optional
            Activation function(s) for the decoder, by default 'LeakyReLU'.
        decoder_dropout : Union[float, List[float]], optional
            Dropout rate(s) for the decoder, by default 0.2.
        decoder_bias : bool, optional
            Whether to use bias in the decoder, by default True.
        decoder_output_bias : bool, optional
            Whether to use bias in the output layer of the decoder, by default True.
        dt : float, optional
            Time step for the SDE solver, by default 0.1.
        adjoint : bool, optional
            Whether to use the adjoint method for the SDE solver, by default False.
        loading_existing : bool, optional
            Whether to load an existing model, by default False.

        Returns
        -------
        None

        Notes
        -----
        This class implements a VAE with fixed potential SDE using PyTorch Lightning.

        Examples
        --------
        >>> model = LightningSDE_VAE_FixedPotential(data_dim=100, latent_dim=20, dt=0.05)
        >>> model.fit(data)
        """
        super().__init__()

        name = self._configure_name(name, loading_existing=loading_existing)

        self.save_hyperparameters()

        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(func=neural_diffeqs.PotentialSDE, kwargs=locals())
        self._configure_lightning_model(kwargs=locals())

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

    def __repr__(self) -> Literal['LightningSDE-VAE-FixedPotential']:
        return "LightningSDE-VAE-FixedPotential"
