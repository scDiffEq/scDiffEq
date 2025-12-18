# -- import packages: ---------------------------------------------------------
import neural_diffeqs
import torch

# -- import local dependencies: -----------------------------------------------
from . import base, mix_ins

# -- set type hints: ----------------------------------------------------------
from typing import List, Literal, Optional, Union

# -- DiffEq: ------------------------------------------------------------------
class LightningODE_FixedPotential(
    mix_ins.BaseForwardMixIn,
    mix_ins.PotentialMixIn,
    base.BaseLightningDiffEq,
):
    """LightningODE-FixedPotential"""
    def __init__(
        self,
        latent_dim: int = 50,
        name: Optional[str] = None,
        dt: float = 0.1,
        mu_hidden: Union[List[int], int] = [400, 400],
        mu_activation: Union[str, List[str]] = "LeakyReLU",
        mu_dropout: Union[float, List[float]] = 0.2,
        mu_bias: bool = True,
        mu_output_bias: bool = True,
        mu_n_augment: int = 0,
        train_lr=1e-4,
        train_optimizer=torch.optim.RMSprop,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        train_step_size: int = 10,
        backend: str = "auto",
        adjoint=False,
        # -- other: -----------------------------------------------------------
        loading_existing: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        LightningODE_FixedPotential

        Parameters:
        -----------
        latent_dim : int, optional
            Dimensionality of the latent space, by default 50
        name : str, optional
            Name of the model, by default None
        dt : float, optional
            Time step for the ODE solver, by default 0.1
        mu_hidden : Union[List[int], int], optional
            Hidden layer sizes for the neural network, by default [400, 400]
        mu_activation : Union[str, List[str]], optional
            Activation function(s) for the neural network, by default "LeakyReLU"
        mu_dropout : Union[float, List[float]], optional
            Dropout rate(s) for the neural network, by default 0.2
        mu_bias : bool, optional
            Whether to use bias in the neural network, by default True
        mu_output_bias : bool, optional
            Whether to use bias in the output layer of the neural network, by default True
        mu_n_augment : int, optional
            Number of augmentations for the neural network, by default 0
        train_lr : float, optional
            Learning rate for training, by default 1e-4
        train_optimizer : torch.optim.Optimizer, optional
            Optimizer for training, by default torch.optim.RMSprop
        train_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler for training, by default torch.optim.lr_scheduler.StepLR
        train_step_size : int, optional
            Step size for the learning rate scheduler, by default 10
        backend : str, optional
            Backend for the ODE solver, by default "auto"
        adjoint : bool, optional
            Whether to use the adjoint method for the ODE solver, by default False
        loading_existing : bool, optional
            Whether to load an existing model, by default False

        Returns:
        --------
        None

        Notes:
        ------
        This class implements a fixed potential ODE using PyTorch Lightning.

        Examples:
        ---------
        >>> model = LightningODE_FixedPotential(latent_dim=20, dt=0.05)
        >>> model.fit(data)
        """
        super().__init__()

        name = self._configure_name(name, loading_existing=loading_existing)

        self.save_hyperparameters()

        self._configure_torch_modules(func=neural_diffeqs.PotentialODE, kwargs=locals())
        self._configure_lightning_model(kwargs=locals())

    def __repr__(self) -> Literal['LightningODE-FixedPotential']:
        return "LightningODE-FixedPotential"
