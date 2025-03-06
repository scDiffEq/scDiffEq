# -- import packages: ---------------------------------------------------------
import neural_diffeqs
import torch

# -- import local dependencies: -----------------------------------------------
from . import base, mix_ins

# -- set type hints: ----------------------------------------------------------
from typing import Dict, List, Literal, Optional, Union

# -- lightning model: ---------------------------------------------------------
class LightningSDE_RegularizedVelocityRatio(
    mix_ins.RegularizedVelocityRatioMixIn,
    base.BaseLightningDiffEq,
):
    def __init__(
        self,
        latent_dim: int = 50,
        velocity_ratio_params: Dict[str, Union[float, bool]] = {
            "target": 1,
            "enforce": 0,  # zero to disable
            "method": "square",  # abs -> calls torch.abs or torch.square
        },
        name: Optional[str] = None,
        mu_hidden: Union[List[int], int] = [512, 512],
        sigma_hidden: Union[List[int], int] = [32, 32],
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
        train_lr: float = 1e-4,
        train_optimizer: torch.optim.Optimizer = torch.optim.RMSprop,
        train_scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.StepLR,
        train_step_size: int = 10,
        dt: float = 0.1,
        adjoint: bool = False,
        backend: str = "auto",
        loading_existing: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        LightningSDE model accessed as model.DiffEq

        Parameters
        ----------
        latent_dim : int, optional
            Number of latent dimensions over which SDE should be parameterized. Default is 50.
        velocity_ratio_params : Dict[str, Union[float, bool]], optional
            Parameters for velocity ratio regularization. Default is {"target": 1, "enforce": 0, "method": "square"}.
        name : Optional[str], optional
            Model name used during project saving. Default is None.
        mu_hidden : Union[List[int], int], optional
            Hidden layer sizes for the drift function. Default is [512, 512].
        sigma_hidden : Union[List[int], int], optional
            Hidden layer sizes for the diffusion function. Default is [32, 32].
        mu_activation : Union[str, List[str]], optional
            Activation function for the drift function. Default is 'LeakyReLU'.
        sigma_activation : Union[str, List[str]], optional
            Activation function for the diffusion function. Default is 'LeakyReLU'.
        mu_dropout : Union[float, List[float]], optional
            Dropout rate for the drift function. Default is 0.1.
        sigma_dropout : Union[float, List[float]], optional
            Dropout rate for the diffusion function. Default is 0.1.
        mu_bias : bool, optional
            Whether to use bias in the drift function. Default is True.
        sigma_bias : List[bool], optional
            Whether to use bias in the diffusion function. Default is True.
        mu_output_bias : bool, optional
            Whether to use output bias in the drift function. Default is True.
        sigma_output_bias : bool, optional
            Whether to use output bias in the diffusion function. Default is True.
        mu_n_augment : int, optional
            Number of augmentations for the drift function. Default is 0.
        sigma_n_augment : int, optional
            Number of augmentations for the diffusion function. Default is 0.
        sde_type : str, optional
            Type of SDE. Default is 'ito'.
        noise_type : str, optional
            Type of noise for SDE. Default is 'general'.
        brownian_dim : int, optional
            Dimension of the Brownian motion. Default is 1.
        coef_drift : float, optional
            Coefficient for the drift term. Default is 1.0.
        coef_diffusion : float, optional
            Coefficient for the diffusion term. Default is 1.0.
        train_lr : float, optional
            Learning rate for training. Default is 1e-4.
        train_optimizer : torch.optim.Optimizer, optional
            Optimizer for training. Default is torch.optim.RMSprop.
        train_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler for training. Default is torch.optim.lr_scheduler.StepLR.
        train_step_size : int, optional
            Step size for the learning rate scheduler. Default is 10.
        dt : float, optional
            Time step for the SDE solver. Default is 0.1.
        adjoint : bool, optional
            Whether to use the adjoint method for the SDE solver. Default is False.
        backend : str, optional
            Backend for the SDE solver. Default is 'auto'.
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
        self._configure_torch_modules(func=neural_diffeqs.NeuralSDE, kwargs=locals())
        self._configure_lightning_model(kwargs=locals())

    def __repr__(self) -> Literal["LightningSDE-RegularizedVelocityRatio"]:
        return "LightningSDE-RegularizedVelocityRatio"
