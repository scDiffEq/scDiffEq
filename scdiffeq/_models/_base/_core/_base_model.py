
__module_name__ = "_base_model.py"
__doc__ = """To-do."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# specify version: -----------------------------------------------------------------------
__version__ = "0.0.44"


# -- import packages: --------------------------------------------------------------------
from pytorch_lightning import LightningModule, loggers
from neural_diffeqs import NeuralSDE, NeuralODE
from torch_composer import TorchNet
import torch


# -- import local dependencies: ----------------------------------------------------------
from ._batch_forward import BatchForward
from ._sinkhorn_divergence import SinkhornDivergence
# from . import _base_ancilliary as base
# from . import _lightning_callbacks as cbs


# -- Lightning base: ---------------------------------------------------------------------
class BaseLightningModel(LightningModule):
    """Base pytorch-lightning model wrapped / trained within models.scDiffEq"""

    def __init__(
        self,
        func: [NeuralSDE, NeuralODE, TorchNet],
        dt=0.1,
        optimizer_kwargs={"lr": 1e-4},
        scheduler_kwargs={"step_size": 20, "gamma": 0.1},
        ignore_t0=True,
    ):
        """To-do: docs"""
        super(BaseLightningModel, self).__init__()
        self.__register_inputs__(func, dt, ignore_t0)

    def __register_inputs__(self, func, dt, ignore_t0):
        """To-do: docs"""

        self.func = func
        self.dt = dt
        self.save_hyperparameters(ignore=["func"])
        self.__configure_forward_step__(ignore_t0)

    def __configure_forward_step__(self, ignore_t0):
        """To-Do: docs"""
        forward_step = BatchForward(self.func, loss_function = SinkhornDivergence)
        setattr(self, "forward", getattr(forward_step, "__call__"))

    def configure_optimizers(self):
        """To-Do: docs"""
        optimizer = torch.optim.RMSprop(
            self.parameters(), **self.hparams["optimizer_kwargs"]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, **self.hparams["scheduler_kwargs"]
        )
        return [optimizer], [scheduler]


# BaseModel: -----------------------------------------------------------------------------
class BaseModel(BaseLightningModel):
    def __init__(self, func, **kwargs):
        super(BaseModel, self).__init__(func, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.forward(self, batch, stage="train", dt=self.dt)

    def validation_step(self, batch, batch_idx):
        return self.forward(self, batch, stage="val", dt=self.dt)

    def test_step(self, batch, batch_idx):
        return self.forward(self, batch, stage="test", dt=self.dt)

    def predict_step(self, batch, batch_idx):
        return self.forward(self, batch, stage="predict", dt=self.dt)
