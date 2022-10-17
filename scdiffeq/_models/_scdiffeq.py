
__module_name__ = "_scDiffEq.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# import packages: ------------------------------------------------------------
from pytorch_lightning import LightningDataModule
import anndata
import torch

from neural_diffeqs import NeuralODE, NeuralSDE
from torch_composer import TorchNet

from ._core._base_model import BaseModel

# MAIN MODULE CLASS: scDiffEq Model -------------------------------------------

# -----------------------------------------------------------------------------
class scDiffEq(BaseModel):
    
    def __init__(self,
                 adata: anndata.AnnData = None,
                 DataModule: LightningDataModule=None,
                 func:[NeuralODE, NeuralSDE, torch.nn.Module] = None,
                ):
        
        super(scDiffEq, self).__init__(func)
        
        
    def fit(self):
        
        ...
        
        