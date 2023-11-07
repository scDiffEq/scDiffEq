
# -- import packages: ----------------------------------------------------------
from neural_diffeqs import PotentialODE
import torch


# -- import local dependencies: ------------------------------------------------
from . import base, mix_ins


# -- set typing: ---------------------------------------------------------------
from typing import Optional, Union, List
from ... import __version__


# -- DiffEq: -------------------------------------------------------------------
class LightningODE_FixedPotential(
    mix_ins.BaseForwardMixIn,
    mix_ins.PotentialMixIn,
    base.BaseLightningDiffEq,
):
    def __init__(
        self,
        latent_dim: int = 50,
        name: Optional[str] = None,
        dt=0.1,
        mu_hidden: Union[List[int], int] = [400, 400],
        mu_activation: Union[str, List[str]] = 'LeakyReLU',
        mu_dropout: Union[float, List[float]] = 0.2,
        mu_bias: bool = True,
        mu_output_bias: bool = True,
        mu_n_augment: int = 0,
        train_lr=1e-4,
        train_optimizer=torch.optim.RMSprop,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        train_step_size=10,
        backend = "auto",
        
        adjoint=False,
        version = __version__,

        *args,
        **kwargs,
    )->None:
        """
        LightningODE_FixedPotential
        
        Parameters:
        -----------
        
        Returns:
        --------
        
        Notes:
        ------
        
        Examples:
        ---------
        """
        super().__init__()
        
        name = self._configure_name(name)

        self.save_hyperparameters()

        self._configure_torch_modules(func=PotentialODE, kwargs=locals())
        self._configure_lightning_model(kwargs = locals())

    def __repr__(self) -> str:
        return "LightningODE-FixedPotential"
