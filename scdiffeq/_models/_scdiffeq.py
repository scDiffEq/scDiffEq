
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


# -- version: ----------------------------------------------------------------------------
__version__ = "0.0.44"


# -- import packages: --------------------------------------------------------------------
from pytorch_lightning import LightningDataModule
from neural_diffeqs import NeuralODE, NeuralSDE
from torch_nets import TorchNet
import anndata
import torch
import os


# -- import local dependencies: ----------------------------------------------------------
from . import _base as base


# -- Focus of this module: scDiffEq model: -----------------------------------------------
class scDiffEq(base.BaseModel):
    
    def __init__(self,
                 adata: anndata.AnnData = None,
                 DataModule: LightningDataModule=None,
                 func:[NeuralODE, NeuralSDE, torch.nn.Module] = None,
                 time_key="Time point",
                 use_key="X_pca",
                 batch_size: int = 2000,
                 num_workers: int = os.cpu_count(),
                 optimizer_kwargs: dict = {"lr": 1e-4},
                 scheduler_kwargs: dict = {"step_size": 20, "gamma": 0.1},
                 ignore_t0: bool = True,
                 dt: float = 0.1,
                 percent_val: float = 0.2, 
                 devices: int = torch.cuda.device_count(),
                 model_save_dir='scDiffEq_model',
                 log_name='lightning_logs',
                 version=None,
                 prefix='',
                 flush_logs_every_n_steps=5,
                 max_epochs=1500,
                 log_every_n_steps=1,
                 reload_dataloaders_every_n_epochs=5,
                 report_kwargs=False,
                 **kwargs,
                 # TODO: ENCODER/DECODER KWARGS
                ):
       
    
        """
        Primary user-facing model.
        
        Parameters:
        -----------
        adata
            type: anndata.AnnData
            default: None


        DataModule
            For data already processed and formatted into a LightningDataModule from pytorch-lightning.
            Users familiar with pytorch-lightning may wish to preformat their data in this way. If used,
            steps to generate a LightningDataModule from AnnData are bypassed.
            type: pytorch_lightning.LightningDataModule
            default: None            

        func
            type: [NeuralODE, NeuralSDE, torch.nn.Module]
            default: None
        
        Keyword Arguments:
        ------------------


        Returns:
        --------
        None
        
        Notes:
        ------

        """
        
        if not func:
            func = base.default_NeuralSDE(state_size=adata.obsm[use_key].shape[1])
        
        super(scDiffEq, self)
        self.__configure__(report_kwargs, locals())
