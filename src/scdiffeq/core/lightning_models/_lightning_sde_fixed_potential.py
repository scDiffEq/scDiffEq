# -- import packages: ---------------------------------------------------------
import neural_diffeqs
import torch

# -- import local dependencies: -----------------------------------------------
from . import base, mix_ins

# -- set type hints: ----------------------------------------------------------
from typing import List, Literal, Optional, Union

# -- lightning model: ---------------------------------------------------------
class LightningSDE_FixedPotential(
    mix_ins.PotentialMixIn,
    mix_ins.BaseForwardMixIn,
    base.BaseLightningDiffEq,
):
    def __init__(
        self,
        latent_dim: int = 50,
        name: Optional[str] = None,
        mu_hidden: Union[List[int], int] = [2000, 2000],
        sigma_hidden: Union[List[int], int] = [800, 800],
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
        train_lr: float = 1e-4,
        train_optimizer=torch.optim.RMSprop,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        train_step_size: int = 10,
        dt: float = 0.1,
        adjoint=False,
        backend="auto",
        loading_existing: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        LightningSDE_FixedPotential

        Parameters
        ----------
        latent_dim : int, optional
            Dimensionality of the latent space, by default 50.
        name : str, optional
            Name of the model, by default None.
        mu_hidden : Union[List[int], int], optional
            Hidden layer sizes for the drift neural network, by default [2000, 2000].
        sigma_hidden : Union[List[int], int], optional
            Hidden layer sizes for the diffusion neural network, by default [800, 800].
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
        train_lr : float, optional
            Learning rate for training, by default 1e-4.
        train_optimizer : torch.optim.Optimizer, optional
            Optimizer for training, by default torch.optim.RMSprop.
        train_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler for training, by default torch.optim.lr_scheduler.StepLR.
        train_step_size : int, optional
            Step size for the learning rate scheduler, by default 10.
        dt : float, optional
            Time step for the SDE solver, by default 0.1.
        adjoint : bool, optional
            Whether to use the adjoint method for the SDE solver, by default False.
        backend : str, optional
            Backend for the SDE solver, by default "auto".
        loading_existing : bool, optional
            Whether to load an existing model, by default False.

        Returns
        -------
        None

        Notes
        -----
        This class implements a fixed potential SDE using PyTorch Lightning.

        Examples
        --------
        >>> model = LightningSDE_FixedPotential(latent_dim=20, dt=0.05)
        >>> model.fit(data)
        """
        super().__init__()

        name = self._configure_name(name, loading_existing=loading_existing)

        self.save_hyperparameters()

        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(func=neural_diffeqs.PotentialSDE, kwargs=locals())
        self._configure_lightning_model(kwargs=locals())

    def __repr__(self) -> Literal['LightningSDE-FixedPotential']:
        return "LightningSDE-FixedPotential"
