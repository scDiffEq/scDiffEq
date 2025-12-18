
# -- import packages: ----------------------------------------------------------
import torch_nets
import lightning
import torch
import ABCParse

# -- import local dependencies: ------------------------------------------------
from ..base._batch_processor import BatchProcessor
from ..base._sinkhorn_divergence import SinkhornDivergence

from ... import utils


# -- set typing: ---------------------------------------------------------------
from typing import Union, List


# -- MixIn class: --------------------------------------------------------------        
class VAEMixIn(object):
    """Mixin class for Variational Autoencoder (VAE) operations.

    This class provides methods for configuring encoder and decoder modules,
    performing forward pass, and optimization steps including pre-training.

    Attributes:
        reconstruction_loss (torch.nn.Module): Reconstruction loss function.

    Methods:
        __init__: Initialize the VAEMixIn object.
        _configure_encoder: Configure the encoder module.
        _configure_decoder: Configure the decoder module.
        _configure_torch_modules: Configure torch modules including encoder, decoder, and differential equation.
        _configure_lightning_model: Configure lightning model with optimizers and schedulers.
        encode: Encode input data into latent space.
        decode: Decode latent space into output data.
        forward: Perform forward pass integrating in latent space.
        pretrain_step: Perform a single pre-training step.
        step: Perform a single optimization step.
        __repr__: Return the representation of the class.

    """
    def __init__(self, *args, **kwargs) -> None:
        
        """
        Initialize the VAEMixIn object.

        Args:
            *args: Variable length arguments.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None

        """
        
        super().__init__()

        # -- loss functions: ---------------------------------------------------
        self.reconstruction_loss = torch.nn.MSELoss(reduction="sum", **kwargs)
        
    def _configure_encoder(self) -> None:
        self.Encoder = torch_nets.Encoder(
            data_dim = self.hparams['data_dim'],
            latent_dim = self.hparams['latent_dim'],
            n_hidden=self.hparams["encoder_n_hidden"],
            power=self.hparams["encoder_power"],
            activation=self.hparams["encoder_activation"],
            dropout=self.hparams["encoder_dropout"],
            bias=self.hparams["encoder_bias"],
            output_bias=self.hparams["encoder_output_bias"],
        )

    def _configure_decoder(self) -> None:
        self.Decoder = torch_nets.Decoder(
            data_dim = self.hparams['data_dim'],
            latent_dim = self.hparams['latent_dim'],
            n_hidden=self.hparams["decoder_n_hidden"],
            power=self.hparams["decoder_power"],
            activation=self.hparams["decoder_activation"],
            dropout=self.hparams["decoder_dropout"],
            bias=self.hparams["decoder_bias"],
            output_bias=self.hparams["decoder_output_bias"],
        )
        
    def _configure_torch_modules(self, func, kwargs) -> None:
        
        kwargs['state_size'] = self.hparams['latent_dim']
        
        self._configure_encoder()
        self.DiffEq = func(**utils.function_kwargs(func, kwargs))
        self._configure_decoder()
#         self._configure_optimizers_schedulers()
        
    def _configure_lightning_model(self, kwargs): # _configure_optimizers_schedulers(self):
        
        """Assumes use of a pre-train step with an independent optimizer, scheduler (i.e., two total, each)"""
        
        pretrain_optimizer = self.hparams['pretrain_optimizer']
        train_optimizer = self.hparams['train_optimizer']
        pretrain_scheduler = self.hparams['pretrain_scheduler']
        train_scheduler = self.hparams['train_scheduler']
        
        VAE_params = list(self.Encoder.parameters()) + list(self.Decoder.parameters())
        self._optimizers = [
            pretrain_optimizer(VAE_params, lr=self.hparams['pretrain_lr']),
            train_optimizer(self.parameters(), lr=self.hparams['train_lr']),
        ]
        
        self._schedulers = [
            pretrain_scheduler(
                optimizer=self._optimizers[0], step_size=self.hparams['pretrain_step_size']
            ),
            train_scheduler(
                    optimizer=self._optimizers[1], step_size=self.hparams['train_step_size']
                ),
        ]
        sinkhorn_kwargs = ABCParse.function_kwargs(func = SinkhornDivergence, kwargs = kwargs)
        self.sinkhorn_divergence = SinkhornDivergence(**sinkhorn_kwargs)
        self.process_batch = BatchProcessor
        self.COMPLETED_EPOCHS = 0
        
        
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        return self.Encoder(X)

    def decode(self, Z: torch.Tensor) -> torch.Tensor:
        return self.Decoder(Z)

    def forward(self, X0, t, logqp=False, **kwargs):
        """Forward step: (0) integrate in latent space"""
        Z_hat = self.integrate(
            Z0=self.encode(X0),
            t=t,
            dt=self.hparams["dt"],
            logqp=logqp,
            **kwargs,
        )
        return self.decode(Z_hat)

    def pretrain_step(self, batch, batch_idx, stage=None):
        """Encode only X0->Z0. Return X0_hat"""
        
        pretrain_optim = self.optimizers()[0]
        pretrain_optim.zero_grad()
        batch = self.process_batch(batch, batch_idx)
        X0_hat = self.decode(self.encode(batch.X0))
        recon_loss = self.reconstruction_loss(X0_hat, batch.X0).sum()
        self.log("pretrain_rl_mse", recon_loss.item())
        
#         self.log_lr()
        
        self.manual_backward(recon_loss)
        pretrain_optim.step()
        pretrain_scheduler = self.lr_schedulers()[0]
        pretrain_scheduler.step()

    def step(self, batch, batch_idx, stage=None):
        
        if stage == "training":
            train_optim = self.optimizers()[1]
            train_optim.zero_grad()

        batch = self.process_batch(batch, batch_idx)
        X_hat = self.forward(batch.X0, batch.t)
        self.sinkhorn_loss = self.compute_sinkhorn_divergence(
            batch.X, X_hat, batch.W, batch.W_hat
        )
#         loss = self.log_sinkhorn_divergence(sinkhorn_loss, t=batch.t, stage=stage)
                
#         self.log_lr()
        
        if stage == "training":
            self.manual_backward(self.sinkhorn_loss) # loss)
            train_optim.step()
            train_scheduler = self.lr_schedulers()[1]
            train_scheduler.step()

    def __repr__(self) -> str:
        return "scDiffEq MixIn: VAEMixIn"
