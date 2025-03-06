# -- import packages: ---------------------------------------------------------
import abc
import ABCParse
import lightning
import torch
import torchsde

# -- import local dependencies: -----------------------------------------------
from ._batch_processor import BatchProcessor
from ._sinkhorn_divergence import SinkhornDivergence

# -- set type hints: ----------------------------------------------------------
from typing import Optional

# -- DiffEq class: ------------------------------------------------------------
class BaseLightningDiffEq(lightning.LightningModule):
    
    """BaseLightningDiffEq: Base class for lightning modules with differential equation solving capabilities.

    This class serves as the base for lightning modules that involve solving differential equations.
    It provides methods for configuring the model, integrating initial value problems (IVPs),
    computing Sinkhorn divergence loss, defining custom steps, and implementing LightningModule methods.

    Attributes:
        None

    Methods:
        __init__: Initialize the BaseLightningDiffEq object.
        _update_lit_diffeq_hparams: Update LightningDifferentialEquation hyperparameters.
        _configure_lightning_model: Configure lightning model with optimizer, scheduler, and additional components.
        _configure_torch_modules: Configure Torch modules for the model.
        PRETRAIN: Property indicating if the model is in pre-training mode.
        _INTEGRATOR: Property returning the integrator function based on hyperparameters.
        integrate: Integrate initial value problems (IVPs) using the specified integrator.
        compute_sinkhorn_divergence: Compute Sinkhorn divergence loss.
        forward: Abstract method for performing forward pass.
        step: Abstract method for performing a step in optimization.
        training_step: LightningModule method for training step.
        validation_step: LightningModule method for validation step.
        test_step: LightningModule method for test step.
        predict_step: LightningModule method for prediction step.
        configure_optimizers: Configure optimizers and schedulers for training.
        __repr__: Return the representation of the class.
        _configure_name: Configure the name of the class.

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseLightningDiffEq object.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            None

        """
        super().__init__()
        
    def _update_lit_diffeq_hparams(self, model_params) -> None:
        """
        Update LightningDifferentialEquation hyperparameters.

        This method updates the hyperparameters of the lightning differential equation model
        based on the provided model parameters.

        Args:
            model_params (dict): Model parameters.

        Returns:
            None

        """
        for key, val in self.hparams.items():
            if key in model_params.keys():
                if val != model_params[key]:
                    self.hparams.update({key: model_params[key]})
        
    # -- setup: ----------------------------------------------------------------
    def _configure_lightning_model(self, kwargs) -> None:
        
        """
        Configure lightning model with optimizer, scheduler, and additional components.

        This method configures the lightning model with optimizer, scheduler, Sinkhorn divergence,
        batch processor, and other components. It assumes there is no pre-train step, which results
        in a single optimizer, scheduler.

        Args:
            kwargs (dict): Additional keyword arguments.

        Returns:
            None

        """
        
        optimizer = self.hparams['train_optimizer']
        scheduler = self.hparams['train_scheduler']

        self._optimizers = [optimizer(self.parameters(), lr=self.hparams['train_lr'])]
        self._schedulers = [
            scheduler(
                optimizer=self._optimizers[0],
                step_size=self.hparams['train_step_size']),
        ]
        sinkhorn_kwargs = ABCParse.function_kwargs(func = SinkhornDivergence, kwargs = kwargs)
        self.sinkhorn_divergence = SinkhornDivergence(**sinkhorn_kwargs)
        self.process_batch = BatchProcessor
        self.COMPLETED_EPOCHS = 0

    def _configure_torch_modules(self, func, kwargs) -> None:
        """Configure Torch modules for the model.

        This method configures Torch modules for the model based on the provided function and arguments.

        Args:
            func: Function for configuring Torch modules.
            kwargs (dict): Additional keyword arguments.

        Returns:
            None

        """
        kwargs['state_size'] = self.hparams['latent_dim']
        self.DiffEq = func(**ABCParse.function_kwargs(func, kwargs))
        
    @property
    def PRETRAIN(self) -> bool:
        """
        Property indicating if the model is in pre-training mode.

        Returns:
            bool: True if in pre-training mode, False otherwise.

        """
        return False
    
    # -- IVP-solving: ---------------------------------------------------------
    @property
    def _INTEGRATOR(self):
        """Property returning the integrator function based on hyperparameters.

        Returns:
            function: Integrator function.

        """
        if self.hparams["adjoint"]:
            return torchsde.sdeint_adjoint
        return torchsde.sdeint

    def integrate(self, Z0, t, dt, logqp, **kwargs) -> torch.Tensor:
        """Integrate initial value problems (IVPs) using the specified integrator.

        This method integrates initial value problems (IVPs) using the specified integrator.

        Args:
            Z0 (torch.Tensor): Initial latent space.
            t (torch.Tensor): Time steps.
            dt (torch.Tensor): Time step size.
            logqp (torch.Tensor): Logarithmic quantities flag.
            **kwargs: Additional keyword arguments.

        Returns:
            Z_hat (torch.Tensor) predicted (integrated) latent space.

        """
        return self._INTEGRATOR(
            sde=self.DiffEq,
            y0=Z0,
            ts=t,
            dt=dt,
            logqp=logqp,
            **kwargs,
        )

    # -- sinkhorn loss: -------------------------------------------------------
    def compute_sinkhorn_divergence(self, X, X_hat, W, W_hat) -> torch.Tensor:
        """Compute Sinkhorn divergence loss.

        This method computes the Sinkhorn divergence loss between the actual and predicted outputs.

        Args:
            X (tenstorch.Tensoror): Actual outputs.
            X_hat (torch.Tensor): Predicted outputs.
            W (torch.Tensor): Actual weights.
            W_hat (torch.Tensor): Predicted weights.

        Returns:
            tensor: Computed Sinkhorn divergence loss.

        """
        return self.sinkhorn_divergence(
            W_hat.contiguous(), X_hat.contiguous(), W.contiguous(), X.contiguous(), 
        ).requires_grad_()


    # -- custom steps: -------------------------------------------------------------
    @abc.abstractmethod
    def forward(self, Z0, t, **kwargs) -> None:
        """
        Abstract method for performing forward pass. Over-written by inheriting class

        Args:
            Z0 (tensor): Initial latent space.
            t (tensor): Time steps.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        """
        ...

    @abc.abstractmethod
    def step(self, batch, batch_idx, stage=None) -> None:
        """Abstract method for performing a step in optimization.

        Args:
            batch (Batch): Input batch of data.
            batch_idx (int): Index of the batch.
            stage (str): Optional stage information.

        Returns:
            None

        """
        print("WARNING: The base (empty) step is being called from `_base_lightning_diffeq.py`")
        ...

    # -- LightningModule methods: ----------------------------------------------
    def training_step(self, batch, batch_idx, *args, **kwargs) -> None:
        """LightningModule method for training step.

        Args:
            batch (Batch): Input batch of data.
            batch_idx (int): Index of the batch.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None

        """
        return self.step(batch, batch_idx, stage="training")

    def validation_step(self, batch, batch_idx=None, *args, **kwargs) -> None:
        """LightningModule method for validation step.

        Args:
            batch (Batch): Input batch of data.
            batch_idx (int): Index of the batch.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None

        """
        return self.step(batch, batch_idx, stage="validation")

    def test_step(self, batch, batch_idx=None, *args, **kwargs) -> None:
        """LightningModule method for test step.

        Args:
            batch (Batch): Input batch of data.
            batch_idx (int): Index of the batch.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None

        """
        return self.step(batch, batch_idx, stage="test")

    def predict_step(self, batch, batch_idx=None, *args, **kwargs) -> None:
        """LightningModule method for prediction step.

        Args:
            batch (Batch): Input batch of data.
            batch_idx (int): Index of the batch.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None

        """
        return self.step(batch, batch_idx, stage="predict")

    def configure_optimizers(self):
        """Configure optimizers and schedulers for training.

        Returns:
            tuple: Tuple containing optimizers and schedulers.

        """
        return self._optimizers, self._schedulers
    
    def __repr__(self) -> str:
        """Return the representation of the class.

        Returns:
            str: Representation of the class.

        """
        return "LightningDiffEq"
    
    def _configure_name(self, name: Optional[str] = None, delim: Optional[str] = ".", loading_existing: bool = False) -> str:
        """Configure the name of the class.

        Args:
            name (str, optional): Name of the class.
            delim (str, optional): Delimiter for the name.
            loading_existing (bool, optional): Flag indicating if loading existing class.

        Returns:
            str: Configured name of the class.

        """
                
        if loading_existing or name is None:
            return self.__repr__()
        else:
            return f"{self.__repr__()}{delim}{name}"


# -- moved to log callback: ---
#     def log_sinkhorn_divergence(self, sinkhorn_loss, t, stage):
#         for i in range(len(t)):
#             _t = round(t[i].item(), 3)
#             msg = f"sinkhorn_{_t}_{stage}"
#             val = sinkhorn_loss[i]
#             self.log(msg, val)

#         return sinkhorn_loss.sum()

#     def log_lr(self):

#         if not isinstance(self.optimizers(), list):
#             lr = self.optimizers().optimizer.state_dict()["param_groups"][0]["lr"]
#             self.log("opt_param_group_lr", lr)
#         else:
#             for i, opt in enumerate(self.optimizers()):
#                 for j, pg in enumerate(opt.optimizer.state_dict()["param_groups"]):
#                     self.log(f"opt_{i}_param_group_{j}_lr", pg["lr"])

#     def log_total_epochs(self):
#         """Train model N times --> N"""
#         self.log("total_epochs", self.COMPLETED_EPOCHS)
