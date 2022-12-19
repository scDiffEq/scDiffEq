
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
from torchsde import sdeint
import torch


# -- LightningModel: ---------------------------------------------------------------------
class LightningModel(LightningModule):
    """Pytorch-Lightning model trained within scDiffEq"""
    
    def __config__(self, func, lit_config, kwargs):
                
        self.func = func
        self.lit_config = lit_config(params=func.parameters(), **kwargs)
        self.loss_func = self.lit_config.loss_function
        self.dt = self.lit_config.dt
        self.forward = self.lit_config.forward_method

    def __init__(
        self,
        func: [NeuralSDE, NeuralODE, TorchNet] = None,
        lit_config=None,
        **kwargs,
    ):
        """TODO: docs"""

        super(LightningModel, self).__init__()
        
        self.__config__(func, lit_config, kwargs)

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
        
        return [self.lit_config.optimizer], [self.lit_config.lr_scheduler]

