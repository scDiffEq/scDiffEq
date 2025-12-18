# -- import packages: ---------------------------------------------------------
import neural_diffeqs
import torch

# -- import local dependencies: -----------------------------------------------
from . import base, mix_ins

# -- set type hints: ----------------------------------------------------------
from typing import Literal, Optional, Union, List

# -- lightning model: ---------------------------------------------------------
class LightningSDE_VAE_FateBiasAware(
    mix_ins.VAEMixIn,
    mix_ins.FateBiasVAEMixIn,
    mix_ins.PotentialMixIn,
    mix_ins.PreTrainMixIn,
    base.BaseLightningDiffEq,
):
    """LightningSDE-VAE-FateBiasAware"""
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
        # -- sde params: -------------------------------------------------------
        mu_hidden: Union[List[int], int] = [400, 400, 400],
        sigma_hidden: Union[List[int], int] = [400, 400, 400],
        mu_activation: Union[str, List[str]] = "LeakyReLU",
        sigma_activation: Union[str, List[str]] = "LeakyReLU",
        mu_dropout: Union[float, List[float]] = 0.1,
        sigma_dropout: Union[float, List[float]] = 0.1,
        mu_bias: bool = True,
        sigma_bias: List[bool] = True,
        mu_output_bias: bool = True,
        sigma_output_bias: bool = True,
        mu_n_augment: int = 0,
        sigma_n_augment: int = 0,
        sde_type="ito",
        noise_type="general",
        brownian_dim=1,
        coef_drift: float = 1.0,
        coef_diffusion: float = 1.0,
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
        # -- fate bias parameters: ---------------------------------------------
        t0_idx=None,
        kNN_Graph=None,
        fate_bias_csv_path: Optional[str]=None,
        fate_bias_multiplier: float = 1,
        PCA=None,
        loading_existing: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        LightningSDE_VAE_FateBiasAware

        Parameters
        ----------
        latent_dim : int, optional
            Dimensionality of the latent space, by default 50.
        name : str, optional
            Name of the model, by default None.
        mu_hidden : Union[List[int], int], optional
            Hidden layer sizes for the drift neural network, by default [400, 400, 400].
        sigma_hidden : Union[List[int], int], optional
            Hidden layer sizes for the diffusion neural network, by default [400, 400, 400].
        mu_activation : Union[str, List[str]], optional
            Activation function(s) for the drift neural network, by default 'LeakyReLU'.
        sigma_activation : Union[str, List[str]], optional
            Activation function(s) for the diffusion neural network, by default 'LeakyReLU'.
        mu_dropout : Union[float, List[float]], optional
            Dropout rate(s) for the drift neural network, by default 0.1.
        sigma_dropout : Union[float, List[float]], optional
            Dropout rate(s) for the diffusion neural network, by default 0.1.
        mu_bias : bool, optional
            Whether to use bias in the drift neural network, by default True.
        sigma_bias : List[bool], optional
            Whether to use bias in the diffusion neural network, by default True.
        mu_output_bias : bool, optional
            Whether to use bias in the output layer of the drift neural network, by default True.
        sigma_output_bias : bool, optional
            Whether to use bias in the output layer of the diffusion neural network, by default True.
        mu_n_augment : int, optional
            Number of augmentations for the drift neural network, by default 0.
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
        t0_idx : int, optional
            Initial time index for fate bias, by default None.
        kNN_Graph : object, optional
            k-Nearest Neighbors graph for fate bias, by default None.
        fate_bias_csv_path : str, optional
            Path to the CSV file containing fate bias information, by default None.
        fate_bias_multiplier : float, optional
            Multiplier for fate bias, by default 1.
        PCA : object, optional
            PCA object for dimensionality reduction, by default None.
        loading_existing : bool, optional
            Whether to load an existing model, by default False.

        Returns
        -------
        None

        Notes
        -----
        This class implements a VAE with fate bias aware SDE using PyTorch Lightning.

        Examples
        --------
        >>> model = LightningSDE_VAE_FateBiasAware(latent_dim=50, dt=0.1)
        >>> model.fit(data)
        """
        super().__init__()

        name = self._configure_name(name, loading_existing=loading_existing)

        self.save_hyperparameters(ignore=["kNN_Graph"])

        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(func=neural_diffeqs.NeuralSDE, kwargs=locals())
        self._configure_lightning_model(kwargs=locals())

        self._configure_fate(
            graph=kNN_Graph,
            csv_path=fate_bias_csv_path,
            t0_idx=t0_idx,
            fate_bias_multiplier=fate_bias_multiplier,
            PCA=PCA,
        )

    def __repr__(self) -> Literal['LightningSDE-VAE-FateBiasAware']:
        return "LightningSDE-VAE-FateBiasAware"
