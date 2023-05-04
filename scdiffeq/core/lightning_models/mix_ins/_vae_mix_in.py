
# -- import packages: ----------------------------------------------------------
import torch_nets
import lightning
import torch


from ... import utils

# -- set typing: ---------------------------------------------------------------
from typing import Union, List
NoneType = type(None)


# -- MixIn class: --------------------------------------------------------------        
#     def _configure_pretrain_optimizers(
#         self,
#         pretrain_lr=1e-3,
#         pretrain_step_size=10,
#         optimizer=torch.optim.Adam,
#         scheduler=torch.optim.lr_scheduler.StepLR,
#     ):

#         pretrain_optimizer = optimizer(
#             list(self.Encoder.parameters()) + list(self.Decoder.parameters()),
#             lr=pretrain_lr,
#         )

#         pretrain_scheduler = torch.optim.lr_scheduler.StepLR(
#             pretrain_optimizer, step_size=pretrain_step_size
#         )
#     def _configure_pretrain(
#         self,
#         pretrain_epochs=0,
#         pretrain_lr=1e-3,
#         pretrain_step_size=10,
#         optimizer=torch.optim.Adam,
#         scheduler=torch.optim.lr_scheduler.StepLR,
#     ):
#         self.pretrain_epochs = pretrain_epochs
        
#         # -- Reconstruction Loss Function: -------------------------------------
#         self.MSE_RL = torch.nn.MSELoss(reduction="sum")

        
#         self._configure_pretrain_optimizers(
#             pretrain_lr=pretrain_lr,
#             pretrain_step_size=pretrain_step_size,
#             optimizer=optimizer,
#             scheduler=scheduler,
#         )
        
#         # -- other necessities: ------------------------------------------------
#         self.automatic_optimization = False
        
#     def configure(
#         self,
#         data_dim: int,
#         latent_dim: int,
#         pretrain_epochs=0,
#         pretrain_lr=1e-3,
#         pretrain_step_size=10,
#         optimizer=torch.optim.Adam,
#         scheduler=torch.optim.lr_scheduler.StepLR,
#         encoder_n_hidden: int = 3,
#         encoder_power: float = 2,
#         encoder_activation: Union[str, List[str]] = "LeakyReLU",
#         encoder_dropout: Union[float, List[float]] = 0.2,
#         encoder_bias: bool = True,
#         encoder_output_bias: bool = True,
#         encoder: Union[NoneType, torch.nn.Sequential] = None,
#         decoder_n_hidden: int = 3,
#         decoder_power: float = 2,
#         decoder_activation: Union[str, List[str]] = "LeakyReLU",
#         decoder_dropout: Union[float, List[float]] = 0.2,
#         decoder_bias: bool = True,
#         decoder_output_bias: bool = True,
#         decoder: Union[NoneType, torch.nn.Sequential] = None,
#         mu_hidden: Union[List[int], int] = [2000, 2000],
#         sigma_hidden: Union[List[int], int] = [400, 400],
#         mu_activation: Union[str, List[str]] = 'LeakyReLU',
#         sigma_activation: Union[str, List[str]] = 'LeakyReLU',
#         mu_dropout: Union[float, List[float]] = 0.2,
#         sigma_dropout: Union[float, List[float]] = 0.2,
#         mu_bias: bool = True,
#         sigma_bias: List[bool] = True,
#         mu_output_bias: bool = True,
#         sigma_output_bias: bool = True,
#         mu_n_augment: int = 0,
#         sigma_n_augment: int = 0,
#         sde_type='ito',
#         noise_type='general',
#         brownian_dim=1,
#         coef_drift: float = 1.0,
#         coef_diffusion: float = 1.0,
#         coef_prior_drift: float = 1.0,
#     ):
        
#         KWARGS = {}
        
#         KWARGS["Encoder"] = utils.function_kwargs(func=torch_nets.Encoder, kwargs=locals())
#         KWARGS["SDE"] = utils.function_kwargs(func=self._configure_sde, kwargs=locals())
#         KWARGS["Decoder"] = utils.function_kwargs(func=torch_nets.Decoder, kwargs=locals())
        
#         if isinstance(encoder, NoneType):
#             self.encoder = torch_nets.Encoder(**KWARGS['encoder'])
#         else:
#             self.encoder = encoder
            
#         self._configure_sde()
        
#         if isinstance(decoder, NoneType):
#             decoder = torch_nets.Decoder(**KWARGS['decoder'])
#         else:
#             self.decoder = decoder
        
#         self._configure_SDE(
#             state_size=self.hparams["latent_dim"],
#             **utils.function_kwargs(
#                 func=self._configure_SDE,
#                 kwargs=locals(),
#                 ignore=['state_size'],
#         )

#         self._configure_pretrain(
#             **utils.function_kwargs(
#                 func=self._configure_pretrain,
#                 kwargs=locals(),
#             )

#         self._configure_integrator()


class VAEMixIn(object):
    def __init__(self, *args, **kwargs):
        super().__init__()

        # -- loss functions: ---------------------------------------------------
        self.reconstruction_loss = torch.nn.MSELoss(reduction="sum", **kwargs)
        
    def _configure_encoder(self):
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

    def _configure_decoder(self):
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
        
    def _configure_torch_modules(self, func, kwargs):
        
        kwargs['state_size'] = self.hparams['latent_dim']
        
        self._configure_encoder()
        self.func = func(**utils.function_kwargs(func, kwargs))
        self._configure_decoder()
        
    def _configure_optimizers_schedulers(self):
        
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
        
        self.manual_backward(recon_loss)
        pretrain_optim.step()

    def __repr__(self):
        return "scDiffEq MixIn: VAEMixIn"
