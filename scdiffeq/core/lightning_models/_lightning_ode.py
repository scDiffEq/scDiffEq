
# -- import packages: ----------------------------------------------------------
from neural_diffeqs import NeuralODE
import torch


# -- import local dependencies: ------------------------------------------------
from . import base, mix_ins


from typing import Optional, Union, List
from ... import __version__


# -- lightning model: ----------------------------------------------------------
class LightningODE(
    mix_ins.BaseForwardMixIn,
    base.BaseLightningDiffEq,
):
    def __init__(
        self,
        # -- ode params: -------------------------------------------------------
        latent_dim: int = 50,
        name: Optional[str] = None,
        mu_hidden: Union[List[int], int] = [2000, 2000],
        mu_activation: Union[str, List[str]] = 'LeakyReLU',
        mu_dropout: Union[float, List[float]] = 0.2,
        mu_bias: bool = True,
        mu_output_bias: bool = True,
        mu_n_augment: int = 0,
        sde_type='ito',
        noise_type='general',
        backend = "auto",
        
        # -- general params: ---------------------------------------------------
        train_lr: float = 1e-4,
        train_optimizer = torch.optim.RMSprop,
        train_scheduler = torch.optim.lr_scheduler.StepLR,
        train_step_size: int = 10,
        dt: float = 0.1,
        adjoint=False,
        version = __version__,
        *args,
        **kwargs,
    ):
        super().__init__()
        
        name = self._configure_name(name)

        self.save_hyperparameters()
        
        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(func=NeuralODE, kwargs=locals())
        self._configure_lightning_model(kwargs = locals())

    def __repr__(self):
        return "LightningODE"
