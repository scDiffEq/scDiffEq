# -- import packages: ---------------------------------------------------------
import neural_diffeqs
import torch

# -- import local dependencies: -----------------------------------------------
from . import mix_ins
from . import base

# -- set type hints: ----------------------------------------------------------
from typing import List, Literal, Optional, Union

# -- lightning model: ---------------------------------------------------------
class LightningSDE_VAE_PriorPotential_FateBiasAware(
    mix_ins.FateBiasVAEMixIn,
    mix_ins.DriftPriorVAEMixIn,
    mix_ins.PotentialMixIn,
    mix_ins.VAEMixIn,
    mix_ins.PreTrainMixIn,
    base.BaseLightningDiffEq,
):
    """LightningSDE-VAE-PriorPotential-FateBiasAware"""
    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 50,
        name: Optional[str] = None,
        train_lr: float = 1e-5,
        pretrain_lr: float = 1e-3,
        pretrain_epochs: int = 100,
        pretrain_optimizer: torch.optim.Optimizer = torch.optim.Adam,
        train_optimizer=torch.optim.RMSprop,
        pretrain_scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.StepLR,
        train_scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.StepLR,
        pretrain_step_size: int = 200,
        train_step_size: int = 10,
        dt: float = 0.1,
        adjoint: bool = False,
        backend: str = "auto",
        # -- sde params: ------------------------------------------------------
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
        brownian_dim: int = 1,
        coef_drift: float = 1.0,
        coef_diffusion: float = 1.0,
        coef_prior_drift: float = 1.0,
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
        # -- fate bias parameters: ---------------------------------------------
        t0_idx=None,
        kNN_Graph=None,
        fate_bias_csv_path: Optional[str] = None,
        fate_bias_multiplier: float = 1,
        PCA=None,
        loading_existing: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        LightningSDE-VAE-PriorPotential-FateBiasAware model accessed as model.DiffEq

        Parameters
        ----------
        latent_dim : int, optional
            Number of latent dimensions over which SDE should be parameterized. Default is 50.
        name : Optional[str], optional
            Model name used during project saving. Default is None.
        mu_hidden : Union[List[int], int], optional
            Hidden layer sizes for the drift function. Default is [400, 400, 400].
        sigma_hidden : Union[List[int], int], optional
            Hidden layer sizes for the diffusion function. Default is [400, 400, 400].
        mu_activation : Union[str, List[str]], optional
            Activation function for the drift function. Default is "LeakyReLU".
        sigma_activation : Union[str, List[str]], optional
            Activation function for the diffusion function. Default is "LeakyReLU".
        mu_dropout : Union[float, List[float]], optional
            Dropout rate for the drift function. Default is 0.1.
        sigma_dropout : Union[float, List[float]], optional
            Dropout rate for the diffusion function. Default is 0.1.
        mu_bias : bool, optional
            Whether to use bias in the drift function. Default is True.
        sigma_bias : List[bool], optional
            Whether to use bias in the diffusion function. Default is True.
        mu_output_bias : bool, optional
            Whether to use bias in the output layer of the drift function. Default is True.
        sigma_output_bias : bool, optional
            Whether to use bias in the output layer of the diffusion function. Default is True.
        mu_n_augment : int, optional
            Number of augmented dimensions for the drift function. Default is 0.
        sigma_n_augment : int, optional
            Number of augmented dimensions for the diffusion function. Default is 0.
        sde_type : str, optional
            Type of SDE (e.g., "ito" or "stratonovich"). Default is "ito".
        noise_type : str, optional
            Type of noise (e.g., "general" or "diagonal"). Default is "general".
        brownian_dim : int, optional
            Dimension of the Brownian motion. Default is 1.
        coef_drift : float, optional
            Coefficient for the drift term. Default is 1.0.
        coef_diffusion : float, optional
            Coefficient for the diffusion term. Default is 1.0.
        coef_prior_drift : float, optional
            Coefficient for the prior drift term. Default is 1.0.
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
            Whether to use bias in the output layer of the encoder. Default is True.
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
            Whether to use bias in the output layer of the decoder. Default is True.
        PCA : optional
            Principal Component Analysis. Default is None.
        loading_existing : bool, optional
            Whether to load an existing model. Default is False.

        Returns
        -------
        None
        """
        super().__init__()

        name = self._configure_name(name, loading_existing=loading_existing)

        self.save_hyperparameters()

        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(
            func=neural_diffeqs.LatentPotentialSDE, kwargs=locals()
        )
        self._configure_lightning_model(kwargs=locals())

        self._configure_fate(
            graph=kNN_Graph,
            csv_path=fate_bias_csv_path,
            t0_idx=t0_idx,
            fate_bias_multiplier=fate_bias_multiplier,
            PCA=PCA,
        )

    def __repr__(self) -> Literal['LightningSDE-VAE-PriorPotential-FateBiasAware']:
        return "LightningSDE-VAE-PriorPotential-FateBiasAware"
