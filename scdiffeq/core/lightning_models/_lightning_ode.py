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
    """LightningODE"""

    def __init__(
        self,
        # -- ode params: -------------------------------------------------------
        latent_dim: int = 50,
        name: Optional[str] = None,
        mu_hidden: Union[List[int], int] = [2000, 2000],
        mu_activation: Union[str, List[str]] = "LeakyReLU",
        mu_dropout: Union[float, List[float]] = 0.2,
        mu_bias: bool = True,
        mu_output_bias: bool = True,
        mu_n_augment: int = 0,
        sde_type="ito",
        noise_type="general",
        backend="auto",
        # -- general params: ---------------------------------------------------
        train_lr: float = 1e-4,
        train_optimizer=torch.optim.RMSprop,
        train_scheduler=torch.optim.lr_scheduler.StepLR,
        train_step_size: int = 10,
        dt: float = 0.1,
        adjoint=False,
        loading_existing: bool = False,
        version=__version__,
        *args,
        **kwargs,
    ):
        """
        LightningODE

        Extended description.

        Parameters
        ----------
        latent_dim : int, optional
            Description. Default is 50.
        name : str, optional
            Description. Default is None.
        mu_hidden : Union[List[int], int], optional
            Description. Default is [2000, 2000].
        mu_activation : Union[str, List[str]], optional
            Description. Default is 'LeakyReLU'.
        mu_dropout : Union[float, List[float]], optional
            Description. Default is 0.2.
        mu_bias : bool, optional
            Description. Default is True.
        mu_output_bias : bool, optional
            Description. Default is True.
        mu_n_augment : int, optional
            Description. Default is 0.
        sde_type : str, optional
            Description. Default is 'ito'.
        noise_type : str, optional
            Description. Default is 'general'.
        backend : str, optional
            Description. Default is "auto".
        train_lr : float, optional
            Description. Default is 1e-4.
        train_optimizer : torch.optim.Optimizer, optional
            Description. Default is torch.optim.RMSprop.
        train_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Description. Default is torch.optim.lr_scheduler.StepLR.
        train_step_size : int, optional
            Description. Default is 10.
        dt : float, optional
            Description. Default is 0.1.
        adjoint : bool, optional
            Description. Default is False.
        version : str, optional
            Description. Default is __version__.
        loading_existing : bool, optional
            Whether to load an existing model. Default is False.
        Returns
        -------
        None
        """
        super().__init__()

        name = self._configure_name(name, loading_existing=loading_existing)

        self.save_hyperparameters()

        # -- torch modules: ----------------------------------------------------
        self._configure_torch_modules(func=NeuralODE, kwargs=locals())
        self._configure_lightning_model(kwargs=locals())

    def __repr__(self) -> str:
        return "LightningODE"
