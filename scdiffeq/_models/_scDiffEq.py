
__module_name__ = "_scDiffEq.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# import packages: ------------------------------------------------------------
from pytorch_lightning import LightningDataModule
from neural_diffeqs import neural_diffeq
import anndata
import torch


from ._core._BaseModel import BaseModel

# MAIN MODULE CLASS: scDiffEq Model -------------------------------------------

# -----------------------------------------------------------------------------
class scDiffEq(BaseModel):
    
    def __init__(self,
                 adata: anndata.AnnData = None,
                 DataModule: LightningDataModule=None,
                 func:[neural_diffeq, torch.nn.Module] = None,
                ):
        
        super(scDiffEq, self).__init__(func)
        
        
    def fit(self):
        
        ...
        
        