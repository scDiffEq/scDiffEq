# -- import packages: ---------------------------------------------------------
import neural_diffeqs
import torch

# -- import local dependencies: -----------------------------------------------
from . import mix_ins
from . import base

# -- set type hints: ----------------------------------------------------------
from typing import List, Literal, Optional, Union


# -- lightning model: ---------------------------------------------------------
class LightningODE_VAE(
    mix_ins.VAEMixIn,
    mix_ins.PreTrainMixIn,
    base.BaseLightningDiffEq,
):
    """LightningODE-VAE"""
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
        # -- ode params: -------------------------------------------------------
        mu_hidden: Union[List[int], int] = [400, 400, 400],
        mu_activation: Union[str, List[str]] = "LeakyReLU",
        mu_dropout: Union[float, List[float]] = 0.1,
        mu_bias: bool = True,
        mu_output_bias: bool = True,
        mu_n_augment: int = 0,
        sde_type: str = "ito",
        noise_type: str = "general",
        # -- encoder parameters: -----------------------------------------------
        encoder_n_hidden: int = 4,
        encoder_power: float = 2,
        encoder_activation: Union[str, List[str]] = "LeakyReLU",
        encoder_dropout: Union[float, List[float]] = 0.2,
        encoder_bias: bool = True,
        encoder_output_bias: bool = True,
        # -- decoder parameters: -----------------------------------------------
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
        LightningODE_VAE

        Extended description.

        Parameters
        ----------
        data_dim : int
            Dimension of the data.
        latent_dim : int, optional
            Dimension of the latent space. Default is 50.
        name : str, optional
            Name of the model. Default is None.
        train_lr : float, optional
            Learning rate for training. Default is 1e-5.
        pretrain_lr : float, optional
            Learning rate for pretraining. Default is 1e-3.
        pretrain_epochs : int, optional
            Number of epochs for pretraining. Default is 100.
        pretrain_optimizer : torch.optim.Optimizer, optional
            Optimizer for pretraining. Default is torch.optim.Adam.
        train_optimizer : torch.optim.Optimizer, optional
            Optimizer for training. Default is torch.optim.RMSprop.
        pretrain_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler for pretraining. Default is torch.optim.lr_scheduler.StepLR.
        train_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler for training. Default is torch.optim.lr_scheduler.StepLR.
        pretrain_step_size : int, optional
            Step size for pretraining scheduler. Default is 200.
        train_step_size : int, optional
            Step size for training scheduler. Default is 10.
        dt : float, optional
            Time step for ODE solver. Default is 0.1.
        adjoint : bool, optional
            Whether to use the adjoint method for ODE solver. Default is False.
        backend : str, optional
            Backend for ODE solver. Default is "auto".
        mu_hidden : Union[List[int], int], optional
            Hidden layer sizes for the ODE function. Default is [400, 400, 400].
        mu_activation : Union[str, List[str]], optional
            Activation function for the ODE function. Default is "LeakyReLU".
        mu_dropout : Union[float, List[float]], optional
            Dropout rate for the ODE function. Default is 0.1.
        mu_bias : bool, optional
            Whether to use bias in the ODE function. Default is True.
        mu_output_bias : bool, optional
            Whether to use output bias in the ODE function. Default is True.
        mu_n_augment : int, optional
            Number of augmentations for the ODE function. Default is 0.
        sde_type : str, optional
            Type of SDE. Default is "ito".
        noise_type : str, optional
            Type of noise for SDE. Default is "general".
        encoder_n_hidden : int, optional
            Number of hidden layers for the encoder. Default is 4.
        encoder_power : float, optional
            Power for the encoder. Default is 2.
        encoder_activation : Union[str, List[str]], optional
            Activation function for the encoder. Default is "LeakyReLU".
        encoder_dropout : Union[float, List[float]], optional
            Dropout rate for the encoder. Default is 0.2.
        encoder_bias : bool, optional
            Whether to use bias in the encoder. Default is True.
        encoder_output_bias : bool, optional
            Whether to use output bias in the encoder. Default is True.
        decoder_n_hidden : int, optional
            Number of hidden layers for the decoder. Default is 4.
        decoder_power : float, optional
            Power for the decoder. Default is 2.
        decoder_activation : Union[str, List[str]], optional
            Activation function for the decoder. Default is "LeakyReLU".
        decoder_dropout : Union[float, List[float]], optional
            Dropout rate for the decoder. Default is 0.2.
        decoder_bias : bool, optional
            Whether to use bias in the decoder. Default is True.
        decoder_output_bias : bool, optional
            Whether to use output bias in the decoder. Default is True.
        loading_existing : bool, optional
            Whether to load an existing model. Default is False.

        Returns
        -------
        None
        """
        super().__init__()

        name = self._configure_name(name, loading_existing=loading_existing)

        self.save_hyperparameters()

        # -- torch modules: ---------------------------------------------------
        self._configure_torch_modules(func=neural_diffeqs.NeuralODE, kwargs=locals())
        self._configure_lightning_model(kwargs=locals())

    def __repr__(self) -> Literal["LightningODE-VAE"]:
        return "LightningODE-VAE"
