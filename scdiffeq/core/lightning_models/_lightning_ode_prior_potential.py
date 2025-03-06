# -- import packages: ---------------------------------------------------------
import neural_diffeqs
import torch

# -- import local dependencies: -----------------------------------------------
from . import base, mix_ins
from .. import utils

# -- set type hints: ----------------------------------------------------------
from typing import Literal, Optional, Union, List

# -- lightning model class: ---------------------------------------------------
class LightningODE_PriorPotential(
    mix_ins.PotentialMixIn,
    mix_ins.DriftPriorMixIn,
    base.BaseLightningDiffEq,
):
    """LightningODE-PriorPotential"""
    def __init__(
        self,
        # -- general params: --------------------------------------------------
        latent_dim: int = 50,
        name: Optional[str] = None,
        train_lr: float = 1e-5,
        train_optimizer: torch.optim.Optimizer = torch.optim.RMSprop,
        train_scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.StepLR,
        train_step_size: int = 10,
        adjoint: bool = False,
        # -- ODE params: ------------------------------------------------------
        coef_diff: float = 0,
        dt: float = 0.1,
        mu_hidden: Union[List[int], int] = [400, 400],
        mu_activation: Union[str, List[str]] = "LeakyReLU",
        mu_dropout: Union[float, List[float]] = 0.2,
        mu_bias: bool = True,
        mu_output_bias: bool = True,
        mu_n_augment: int = 0,
        sde_type: str = "ito",
        noise_type: str = "general",
        backend: str = "auto",
        brownian_dim: int = 1,
        loading_existing: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        LightningODE_PriorPotential

        Parameters:
        -----------
        latent_dim : int, optional
            Dimensionality of the latent space, by default 50
        name : str, optional
            Name of the model, by default None
        train_lr : float, optional
            Learning rate for training, by default 1e-5
        train_optimizer : torch.optim.Optimizer, optional
            Optimizer for training, by default torch.optim.RMSprop
        train_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler for training, by default torch.optim.lr_scheduler.StepLR
        train_step_size : int, optional
            Step size for the learning rate scheduler, by default 10
        adjoint : bool, optional
            Whether to use the adjoint method for the ODE solver, by default False
        coef_diff : float, optional
            Coefficient of diffusion, by default 0
        dt : float, optional
            Time step for the ODE solver, by default 0.1
        mu_hidden : Union[List[int], int], optional
            Hidden layer sizes for the neural network, by default [400, 400]
        mu_activation : Union[str, List[str]], optional
            Activation function(s) for the neural network, by default 'LeakyReLU'
        mu_dropout : Union[float, List[float]], optional
            Dropout rate(s) for the neural network, by default 0.2
        mu_bias : bool, optional
            Whether to use bias in the neural network, by default True
        mu_output_bias : bool, optional
            Whether to use bias in the output layer of the neural network, by default True
        mu_n_augment : int, optional
            Number of augmentations for the neural network, by default 0
        sde_type : str, optional
            Type of stochastic differential equation, by default 'ito'
        noise_type : str, optional
            Type of noise, by default 'general'
        backend : str, optional
            Backend for the ODE solver, by default "auto"
        brownian_dim : int, optional
            Dimensionality of the Brownian motion, by default 1
        loading_existing : bool, optional
            Whether to load an existing model, by default False

        Returns:
        --------
        None

        Notes:
        ------
        This class implements a prior potential ODE using PyTorch Lightning.

        Examples:
        ---------
        >>> model = LightningODE_PriorPotential(latent_dim=20, dt=0.05)
        >>> model.fit(data)
        """
        super().__init__()

        name = self._configure_name(name, loading_existing=loading_existing)

        self.save_hyperparameters()
        self.func = LatentPotentialODE(
            state_size=latent_dim,
            **utils.extract_func_kwargs(func=neural_diffeqs.LatentPotentialODE, kwargs=locals()),
        )
        self._configure_lightning_model(kwargs=locals())

    def forward(self, X0, t, **kwargs):
        """Forward step: (0) integrate in latent space"""
        Z_hat, KL_div = self.integrate(
            Z0=X0, t=t, dt=self.hparams["dt"], logqp=True, **kwargs
        )
        return Z_hat, KL_div

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

    def __repr__(self) -> Literal['LightningODE-PriorPotential']:
        return "LightningODE-PriorPotential"
