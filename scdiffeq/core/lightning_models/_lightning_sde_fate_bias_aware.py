
# -- import packages: ----------------------------------------------------------
from neural_diffeqs import NeuralSDE
import torch


# -- import local dependencies: ------------------------------------------------
from . import base, mix_ins
# from scdiffeq.core.lightning_models import base, mix_ins


from typing import Optional, Union, List
from ... import __version__

# -- lightning model: ----------------------------------------------------------
class LightningSDE_FateBiasAware(
    mix_ins.FateBiasMixIn,
    mix_ins.BaseForwardMixIn,
    base.BaseLightningDiffEq,
):
    def __init__(
        self,
        latent_dim: int = 50,
        name: Optional[str] = None,
        mu_hidden: Union[List[int], int] = [400, 400, 400],
        sigma_hidden: Union[List[int], int] = [400, 400, 400],
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
        t0_idx = None,
        kNN_Graph=None,
        fate_bias_csv_path=None,
        fate_bias_multiplier = 1,
        backend = "auto",
        
        version = __version__,
        
        *args,
        **kwargs,
    ):
        super().__init__()
        
        name = self._configure_name(name)

        self.save_hyperparameters(ignore=['kNN_Graph'])
                        
        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(func=NeuralSDE, kwargs=locals())
        self._configure_lightning_model(kwargs = locals())
        self._configure_fate(
            graph=kNN_Graph,
            csv_path = fate_bias_csv_path,
            t0_idx = t0_idx,
            fate_bias_multiplier = fate_bias_multiplier,
            PCA = None,
        )

    def __repr__(self):
        return "LightningSDE-FateBiasAware"
