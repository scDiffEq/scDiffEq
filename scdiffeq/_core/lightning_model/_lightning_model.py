
__module_name__ = "__init__.py"
__version__ = "0.0.45"
__doc__ = """TODO"""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# -- import packages: --------------------------------------------------------------------
from pytorch_lightning import LightningModule
from neural_diffeqs import NeuralODE, NeuralSDE
from torch_nets import TorchNet
import torch

# -- import local dependencies: ----------------------------------------------------------
from ._sinkhorn_divergence import SinkhornDivergence
from ._batch_forward import BatchForward
from .._config import parser


# -- LightningModel: ---------------------------------------------------------------------
class LightningModel(LightningModule):
    
    """Base pytorch-lightning model wrapped / trained within models.scDiffEq"""
    def __init__(self,
                 func: [NeuralSDE, NeuralODE, TorchNet] = None,
                 dt: float = 0.1,
                 optimizer_kwargs={},
                 scheduler_kwargs={},
                 **kwargs,
                ):
        """TODO: docs"""
        
        super(LightningModel, self).__init__()
        parser(self, locals())
        self.__configure_forward_step__()
        
    def __configure_forward_step__(self, ignore_t0=True):
        # TO-DO: documentation
        forward_step = BatchForward(self.func,
                                    loss_function = SinkhornDivergence,
                                    device = self.device,
                                   )
        setattr(self, "forward", getattr(forward_step, "__call__"))
        setattr(self, "integrator", getattr(forward_step, "integrator"))
        setattr(self, "func_type", getattr(forward_step, "func_type"))

        
    def training_step(self, batch, batch_idx):
        # TO-DO: documentation
        return self.forward(self, batch, stage="train", dt=self.dt)

    def validation_step(self, batch, batch_idx):
        # TO-DO: documentation
        return self.forward(self, batch, stage="val", dt=self.dt)

    def test_step(self, batch, batch_idx):
        # TO-DO: documentation
        return self.forward(self, batch, stage="test", dt=self.dt)

    def predict_step(self, batch, batch_idx):
        # TO-DO: documentation
        return self.forward(self, batch, stage="predict", dt=self.dt)

    def configure_optimizers(self):
        # TO-DO: documentation
        # TO-DO: add optimizer / scheduler config to input args through sdq.models.scDiffEq
        optimizer = torch.optim.RMSprop(
            self.parameters(), **self.optimizer_kwargs # .hparams["optimizer_kwargs"]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, **self.scheduler_kwargs # .hparams[""]
        )
        return [optimizer], [scheduler]
