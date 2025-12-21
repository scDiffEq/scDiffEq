# -- import packages: ---------------------------------------------------------
import neural_diffeqs
import torch

# -- import local dependencies: -----------------------------------------------
from . import mix_ins, base

# -- set type hints: ----------------------------------------------------------
from typing import Literal, Optional, Union, List

# -- lightning model cls: -----------------------------------------------------
class LightningODE_VAE_FixedPotential(
    mix_ins.BaseForwardMixIn,
    base.BaseLightningDiffEq,
    mix_ins.PreTrainMixIn,
    mix_ins.PotentialMixIn,
):
    """LightningODE-VAE-FixedPotential"""
    def __init__(
        self,
        data_dim,
        latent_dim: int = 50,
        name: Optional[str] = None,
        mu_hidden: Union[List[int], int] = [2000, 2000],
        mu_activation: Union[str, List[str]] = "LeakyReLU",
        mu_dropout: Union[float, List[float]] = 0.2,
        mu_bias: bool = True,
        mu_output_bias: bool = True,
        mu_n_augment: int = 0,
        train_lr=1e-5,
        pretrain_lr=1e-3,
        pretrain_epochs=100,
        pretrain_optimizer=torch.optim.Adam,
        train_optimizer=torch.optim.RMSprop,
        pretrain_scheduler=None,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        pretrain_step_size=None,
        train_step_size=10,
        dt: float =0.1,
        adjoint: bool =False,
        backend: str = "auto",
        loading_existing: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        LightningODE_VAE_FixedPotential

        Parameters
        ----------
        data_dim : int
            Dimensionality of the input data.
        latent_dim : int, optional
            Dimensionality of the latent space, by default 50.
        name : str, optional
            Name of the model, by default None.
        mu_hidden : Union[List[int], int], optional
            Hidden layer sizes for the neural network, by default [2000, 2000].
        mu_activation : Union[str, List[str]], optional
            Activation function(s) for the neural network, by default 'LeakyReLU'.
        mu_dropout : Union[float, List[float]], optional
            Dropout rate(s) for the neural network, by default 0.2.
        mu_bias : bool, optional
            Whether to use bias in the neural network, by default True.
        mu_output_bias : bool, optional
            Whether to use bias in the output layer of the neural network, by default True.
        mu_n_augment : int, optional
            Number of augmentations for the neural network, by default 0.
        train_lr : float, optional
            Learning rate for training, by default 1e-5.
        pretrain_lr : float, optional
            Learning rate for pretraining, by default 1e-3.
        pretrain_epochs : int, optional
            Number of epochs for pretraining, by default 100.
        pretrain_optimizer : torch.optim.Optimizer, optional
            Optimizer for pretraining, by default torch.optim.Adam.
        train_optimizer : torch.optim.Optimizer, optional
            Optimizer for training, by default torch.optim.RMSprop.
        pretrain_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler for pretraining, by default None.
        train_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler for training, by default torch.optim.lr_scheduler.StepLR.
        pretrain_step_size : int, optional
            Step size for the pretraining learning rate scheduler, by default None.
        train_step_size : int, optional
            Step size for the training learning rate scheduler, by default 10.
        dt : float, optional
            Time step for the ODE solver, by default 0.1.
        adjoint : bool, optional
            Whether to use the adjoint method for the ODE solver, by default False.
        backend : str, optional
            Backend for the ODE solver, by default "auto".
        loading_existing : bool, optional
            Whether to load an existing model, by default False.

        Returns
        -------
        None

        Notes
        -----
        This class implements a VAE with fixed potential ODE using PyTorch Lightning.

        Examples
        --------
        >>> model = LightningODE_VAE_FixedPotential(data_dim=100, latent_dim=20, dt=0.05)
        >>> model.fit(data)
        """
        super().__init__()

        name = self._configure_name(name, loading_existing=loading_existing)

        self.save_hyperparameters()

        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(func=neural_diffeqs.PotentialODE, kwargs=locals())
        self._configure_lightning_model(kwargs=locals())

    def forward(self, X0, t, **kwargs):
        """Forward step: (0) integrate in latent space"""
        Z0 = self.Encoder(X0)
        Z_hat = self.integrate(Z0=Z0, t=t, dt=self.hparams["dt"], logqp=False, **kwargs)
        return self.Decoder(Z_hat)

    def step(self, batch, batch_idx, stage=None):

        batch = self.process_batch(batch, batch_idx)
        X_hat = self.forward(batch.X0, batch.t)
        sinkhorn_loss = self.compute_sinkhorn_divergence(
            batch.X, X_hat, batch.W, batch.W_hat
        )
        return self.log_sinkhorn_divergence(sinkhorn_loss).sum()

    def pretrain_step(self, batch, batch_idx, stage=None):

        batch = self.process_batch(batch, batch_idx)
        X0_hat = self.Decoder(self.Encoder(batch.X0))
        recon_loss = self.reconstruction_loss(X0_hat, batch.X0).sum()
        self.log("pretrain_rl_mse", recon_loss.item())
        return recon_loss

    def __repr__(self) -> Literal['LightningODE-VAE-FixedPotential']:
        return "LightningODE-VAE-FixedPotential"
