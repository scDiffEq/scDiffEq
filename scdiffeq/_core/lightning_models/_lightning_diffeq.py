
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


# -- local impoirts: ---------------------------------------------------------------------
from ._default_neural_sde import default_NeuralSDE
from ..utils import function_kwargs


# -- LightningModel: ---------------------------------------------------------------------
class LightningDiffEq(LightningModule):
    """Pytorch-Lightning model trained within scDiffEq"""
    
    def __parse__(self, kwargs, ignore=["self", "func", "__class__"]):
        
        self.kwargs = {}
        for key, val in kwargs.items():
            if not key in ignore:
                self.kwargs[key] = val
                if key == "kwargs":
                    for k, v in val.items():
                        self.kwargs[k] = v

    def __init__(
        self,
        func: [NeuralSDE, NeuralODE, TorchNet] = None,
        expand: bool = False,
        dt = 0.1,
        **kwargs,
    ):
        """
        Parameters:
        -----------
        func
        
        lit_config
        
        kwargs
        
        Returns:
        --------
        None
        """

        super(LightningDiffEq, self).__init__()
        self.__parse__(locals())
        self.func = func
        self.expand = expand

    def training_step(self, batch, batch_idx)->dict:
        """
        Wraps "LightningModel.forward", indicating stage as "train".
        
        Parameters:
        -----------
        batch
            type: list
        
        batch_idx
            type: int
        
        Returns:
        --------
        forward_out
            Contains at least "loss" key, required for PyTorch-Lightning backprop.
            type: dict
        """
        return self.forward(self, batch, stage="train")

    def validation_step(self, batch, batch_idx):
        """
        Wraps "LightningModel.forward", indicating stage as "val".
        
        Parameters:
        -----------
        batch
            type: list
        
        batch_idx
            type: int
        
        Returns:
        --------
        forward_out
            Contains at least "loss" key, required for PyTorch-Lightning backprop.
            type: dict
        """
        return self.forward(self, batch, stage="val")

    def test_step(self, batch, batch_idx):
        """
        Wraps "LightningModel.forward", indicating stage as "test".
        
        Parameters:
        -----------
        batch
            type: list
        
        batch_idx
            type: int
        
        Returns:
        --------
        forward_out
            Contains at least "loss" key, required for PyTorch-Lightning backprop.
            type: dict
        """
        return self.forward(self, batch, stage="test")

    def predict_step(self, batch, batch_idx):
        """
        Wraps "LightningModel.forward", indicating stage as "predict".
        
        Parameters:
        -----------
        batch
            type: list
        
        batch_idx
            type: int
        
        Returns:
        --------
        forward_out
            Contains at least "loss" key, required for PyTorch-Lightning backprop.
            type: dict
        """
        return self.forward(self, batch, stage="predict", t=self.t, expand=self.expand)

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
        
        return [self.optimizer], [self.lr_scheduler]
