
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
from ._configure_inputs import InputConfiguration
# from . import _base_ancilliary as base
# from . import _lightning_callbacks as cbs

def set_key_as_attr(self, k, v, hide):

    if hide == "all":
        setattr(self, "_{}".format(k), v)

    elif isinstance(hide, str):
        if k == hide:
            setattr(self, "_{}".format(k), v)

    elif isinstance(hide, list):
        if k in hide:
            setattr(self, "_{}".format(k), v)

    else:
        setattr(self, k, v)
        
# -- Lightning base: ---------------------------------------------------------------------
class BaseLightningModel(LightningModule):
    """Base pytorch-lightning model wrapped / trained within models.scDiffEq"""
    
    def __parse__(self, kwargs, hide=None, ignore=["self", "__class__"]):
        
        for k, v in kwargs.items():
            if not k in ignore:
                set_key_as_attr(self, k, v, hide)
    
    def __configure_forward_step__(self, ignore_t0=True):
        """To-Do: docs"""
        forward_step = BatchForward(self.func, loss_function = SinkhornDivergence())
        setattr(self, "forward", getattr(forward_step, "__call__"))
        setattr(self, "integrator", getattr(forward_step, "integrator"))
        setattr(self, "func_type", getattr(forward_step, "func_type"))

    def configure_optimizers(self):
        """To-Do: docs"""
        optimizer = torch.optim.RMSprop(
            self.parameters(), **self.optimizer_kwargs # .hparams["optimizer_kwargs"]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, **self.scheduler_kwargs # .hparams[""]
        )
        return [optimizer], [scheduler]


# LightningModel: ------------------------------------------------------------------------
class LightningModel(BaseLightningModel):
    """TODO: docs"""
    def __init__(self,
                 func: [NeuralSDE, NeuralODE, TorchNet] = None,
                 dt: float = 0.1,
                 optimizer_kwargs={},
                 scheduler_kwargs={},
                 **kwargs,
                ):
        """TODO: docs"""
        super(LightningModel, self).__init__()
        self.__parse__(locals())
        self.__configure_forward_step__()
        
    def training_step(self, batch, batch_idx):
        return self.forward(self, batch, stage="train", dt=self.dt)

    def validation_step(self, batch, batch_idx):
        return self.forward(self, batch, stage="val", dt=self.dt)

    def test_step(self, batch, batch_idx):
        return self.forward(self, batch, stage="test", dt=self.dt)

    def predict_step(self, batch, batch_idx):
        return self.forward(self, batch, stage="predict", dt=self.dt)
