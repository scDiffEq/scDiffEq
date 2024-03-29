
# -- import packages: ---------------------------------------------------------
import neural_diffeqs
import lightning
import torchsde


# -- set typing: --------------------------------------------------------------
from typing import Union, List


# -- MixIn class: -------------------------------------------------------------
class SDEMixIn(lightning.LightningModule):
    """"""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def _configure_SDE(
        self,
        state_size: int,
        mu_hidden: Union[List[int], int] = [400, 400, 400],
        sigma_hidden: Union[List[int], int] = [50, 50, 50],
        mu_activation: Union[str, List[str]] = "LeakyReLU",
        sigma_activation: Union[str, List[str]] = "LeakyReLU",
        mu_dropout: Union[float, List[float]] = 0.2,
        sigma_dropout: Union[float, List[float]] = 0.2,
        mu_bias: bool = True,
        sigma_bias: List[bool] = True,
        mu_output_bias: bool = True,
        sigma_output_bias: bool = True,
        mu_n_augment: int = 0,
        sigma_n_augment: int = 0,
        sde_type="ito",
        noise_type="general",
        brownian_dim=1,
        coef_drift: float = 1.0,
        coef_diffusion: float = 1.0,
    ):
        self.func = neural_diffeqs.NeuralSDE(
            state_size=state_size,
            mu_hidden=mu_hidden,
            sigma_hidden=sigma_hidden,
            mu_activation=mu_activation,
            sigma_activation=sigma_activation,
            mu_dropout=mu_dropout,
            sigma_dropout=sigma_dropout,
            mu_bias=mu_bias,
            sigma_bias=sigma_bias,
            mu_output_bias=mu_output_bias,
            sigma_output_bias=sigma_output_bias,
            mu_n_augment=mu_n_augment,
            sigma_n_augment=sigma_n_augment,
            sde_type=sde_type,
            noise_type=noise_type,
            brownian_dim=brownian_dim,
            coef_drift=coef_drift,
            coef_diffusion=coef_diffusion,
        )
        
    def _configure_integrator(self, adjoint=False):
        if adjoint:
            self._integrator = torchsde.sdeint_adjoint
        else:
            self._integrator = torchsde.sdeint
            
    def __repr__(self):
        return "scDiffEq MixIn: SDEMixIn"
