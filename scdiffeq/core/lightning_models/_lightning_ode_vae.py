
# -- import packages: ----------------------------------------------------------
from neural_diffeqs import NeuralODE
import torch_nets
import torch


# -- import local dependencies: ------------------------------------------------
from . import mix_ins
from . import base


from typing import Optional, Union, List
from ... import __version__



# -- lightning model: ----------------------------------------------------------
class LightningODE_VAE(
    mix_ins.VAEMixIn,
    mix_ins.PreTrainMixIn,
    base.BaseLightningDiffEq,
):
    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 50,
        name: Optional[str] = None,
        train_lr=1e-5,
        pretrain_lr=1e-3,
        pretrain_epochs=100,
        pretrain_optimizer=torch.optim.Adam,
        train_optimizer=torch.optim.RMSprop,
        pretrain_scheduler=torch.optim.lr_scheduler.StepLR,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        pretrain_step_size=200,
        train_step_size=10,
        dt=0.1,
        adjoint=False,
        backend = "auto",

        # -- ode params: -------------------------------------------------------
        mu_hidden: Union[List[int], int] = [400, 400, 400],
        mu_activation: Union[str, List[str]] = 'LeakyReLU',
        mu_dropout: Union[float, List[float]] = 0.1,
        mu_bias: bool = True,
        mu_output_bias: bool = True,
        mu_n_augment: int = 0,
        sde_type='ito',
        noise_type='general',
        
        # -- encoder parameters: -----------------------------------------------
        encoder_n_hidden: int = 4,
        encoder_power: float = 2,
        encoder_activation: Union[str, List[str]] = 'LeakyReLU',
        encoder_dropout: Union[float, List[float]] = 0.2,
        encoder_bias: bool = True,
        encoder_output_bias: bool = True,
        
        # -- decoder parameters: -----------------------------------------------
        decoder_n_hidden: int = 4,
        decoder_power: float = 2,
        decoder_activation: Union[str, List[str]] = 'LeakyReLU',
        decoder_dropout: Union[float, List[float]] = 0.2,
        decoder_bias: bool = True,
        decoder_output_bias: bool = True,
        version = __version__,
        *args,
        **kwargs,
    ):
        super().__init__()
        
        name = self._configure_name(name)

        self.save_hyperparameters()
        
        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(func = NeuralODE, kwargs=locals())
        self._configure_lightning_model(kwargs = locals())

    def __repr__(self):
        return "LightningODE-VAE"