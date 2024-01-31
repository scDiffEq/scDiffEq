
# -- import packages: ----------------------------------------------------------
from neural_diffeqs import NeuralSDE
import torch


# -- import local dependencies: ------------------------------------------------
from . import base, mix_ins


from typing import Optional, Union, List
from ... import __version__


# -- lightning model: ----------------------------------------------------------
class LightningSDE_RegularizedVelocityRatio(
#     mix_ins.BaseForwardMixIn,
    mix_ins.RegularizedVelocityRatioMixIn,
    base.BaseLightningDiffEq,
):
    def __init__(
        self,
        latent_dim: int = 50,
        velocity_ratio_target: float = 1,
        velocity_ratio_enforce: float = 100,
        disable_velocity_ratio_backprop: bool = False,
        name: Optional[str] = None,
        mu_hidden: Union[List[int], int] = [512, 512],
        sigma_hidden: Union[List[int], int] = [32, 32],
        mu_activation: Union[str, List[str]] = 'LeakyReLU',
        sigma_activation: Union[str, List[str]] = 'LeakyReLU',
        mu_dropout: Union[float, List[float]] = 0.1,
        sigma_dropout: Union[float, List[float]] = 0.1,
        mu_bias: bool = True,
        sigma_bias: List[bool] = True,
        mu_output_bias: bool = True,
        sigma_output_bias: bool = True,
        mu_n_augment: int = 0,
        sigma_n_augment: int = 0,
        sde_type='ito',
        noise_type='general',
        brownian_dim=1,
        coef_drift: float = 1.0,
        coef_diffusion: float = 1.0,
        train_lr=1e-4,
        train_optimizer=torch.optim.RMSprop,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        train_step_size=10,
        dt=0.1,
        adjoint=False,
        backend = "auto",
        version = __version__,
        loading_existing: bool = False,
        *args,
        **kwargs,
    ):
        """LightningSDE model accessed as model.DiffEq
        
        Args:
            latent_dim (int): number of latent dimensions over which SDE should be
            parameterized. **Default** = 50
            
            name (Optional[str]): model name used during project saving. **Default** = None
        
        Returns:
            None
        """
        super().__init__()
                
        name = self._configure_name(name, loading_existing = loading_existing)
        
        self.save_hyperparameters()
        
        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(func=NeuralSDE, kwargs=locals())
        self._configure_lightning_model(kwargs = locals())

    def __repr__(self):
        return "LightningSDE-RegularizedVelocityRatio"
