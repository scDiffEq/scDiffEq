
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



# -- LightningModel: ---------------------------------------------------------------------
class LightningModel(LightningModule):
    """Pytorch-Lightning model trained within scDiffEq"""

    def __init__(
        self,
        func: [NeuralSDE, NeuralODE, TorchNet] = None,
        **kwargs,
    ):
        """TODO: docs"""

        super(LightningModel, self).__init__()
        self.lit_config = LightningModelConfig(params=func.parameters(), **kwargs)

    def training_step(self, batch, batch_idx):
        # TODO: documentation
        return self.forward(self, batch, stage="train")

    def validation_step(self, batch, batch_idx):
        # TODO: documentation
        return self.forward(self, batch, stage="val")

    def test_step(self, batch, batch_idx):
        # TODO: documentation
        return self.forward(self, batch, stage="test")

    def predict_step(self, batch, batch_idx):
        # TODO: documentation
        return self.forward(self, batch, stage="predict")

    def configure_optimizers(self):
        """
        Parameters:
        -----------
        None

        Returns:
        --------
        [optimizer], [scheduler]
            type: list, list

        TODO:
        -----
        (1) add documentation
        (2) add support for multiple optimizers & schedulers (needed or not?)
        """
        return [lit_config.optimizer], [lit_config.lr_scheduler]


    
# class LightningModel(LightningModule):
#     def __init__(self,
#                  func: [NeuralSDE, NeuralODE, TorchNet] = None,
#                  dt: float = 0.1,
#                  optimizer_kwargs={},
#                  scheduler_kwargs={},
#                  **kwargs,
#                 ):        
#         super(LightningModel, self).__init__()
#         parser(self, locals())
#         self.__configure_forward_step__()
        
#     def __configure_forward_step__(self, ignore_t0=True):
#         # TODO: documentation
#         forward_step = BatchForward(self.func,
#                                     loss_function = SinkhornDivergence,
#                                     device = self.device,
#                                    )
#         setattr(self, "forward", getattr(forward_step, "__call__"))
#         setattr(self, "integrator", getattr(forward_step, "integrator"))
#         setattr(self, "func_type", getattr(forward_step, "func_type"))

