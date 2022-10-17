
__module_name__ = "_scdiffeq.py"
__doc__ = """To-do."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# version: -------------------------------------------------------------------------------
__version__ = "0.0.44"


# import packages: -----------------------------------------------------------------------
from pytorch_lightning import LightningDataModule
from neural_diffeqs import NeuralODE, NeuralSDE
from torch_composer import TorchNet
import anndata
import torch


# import local dependencies: -------------------------------------------------------------
from ._base._core._base_model import BaseModel


# Focus of this module: scDiffEq model: --------------------------------------------------
class scDiffEq(BaseModel):
    
    def __init__(self,
                 adata: anndata.AnnData = None,
                 DataModule: LightningDataModule=None,
                 func:[NeuralODE, NeuralSDE, torch.nn.Module] = None,
                ):
        
        super(scDiffEq, self).__init__(func)
        
        
    def fit(self):
        
        ...
        
        