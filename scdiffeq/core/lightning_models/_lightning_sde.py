# -- import packages: ----------------------------------------------------------
from neural_diffeqs import NeuralSDE
import torch


# -- import local dependencies: ------------------------------------------------
from . import base, mix_ins


from typing import Optional, Union, List
from ... import __version__


# -- lightning model: ----------------------------------------------------------
class LightningSDE(
    mix_ins.BaseForwardMixIn,
    base.BaseLightningDiffEq,
):
    def __init__(
        self,
        latent_dim: int = 50,
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
        brownian_dim=1,
        coef_drift: float = 1.0,
        coef_diffusion: float = 1.0,
        train_lr=1e-4,
        train_optimizer=torch.optim.RMSprop,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        train_step_size=10,
        dt=0.1,
        adjoint=False,
        backend="auto",
        version=__version__,
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
        name : Optional[str], optional
            Model name used during project saving. Default is None.
        mu_hidden : Union[List[int], int], optional
            Hidden layer sizes for the drift function. Default is [512, 512].
        sigma_hidden : Union[List[int], int], optional
            Hidden layer sizes for the diffusion function. Default is [32, 32].
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
            Whether to use the adjoint method for backpropagation. Default is False.
        backend : str, optional
            Backend for the SDE solver. Default is "auto".
        version : str, optional
            Version of the model. Default is __version__.
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
        self._configure_torch_modules(func=NeuralSDE, kwargs=locals())
        self._configure_lightning_model(kwargs=locals())

    def __repr__(self):
        return "LightningSDE"
