
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
from torch_composer import TorchNet
import anndata
import torch


# -- import local dependencies: ----------------------------------------------------------
from ._base._core._base_model import BaseModel
from ._base._core._configure_lightning_trainer import configure_lightning_trainer


# -- Focus of this module: scDiffEq model: -----------------------------------------------
class scDiffEq(BaseModel):
    
    def __init__(self,
                 adata: anndata.AnnData = None,
                 DataModule: LightningDataModule=None,
                 func:[NeuralODE, NeuralSDE, torch.nn.Module] = None,
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
        Using the following logic, we need to implement a few steps:
        ------------------------------------------------------------
        # adata/use_key -> dataset -> func
        # to get data_dim, need adata and use_key or torch_dataset
        # to pass func, need data_dim
        # need to pass func here
        
        
        ------------------------------------------------------------
        if adata:
            pass to torch_adata.AnnDataset and require use_key
            -> pass to DataModule to get baseline split. requires test/train cols unless the user just wants
            to fit to the whole dataset
            -> define func using data_dim from the DataModule
            
        if DataModule and func:
            nothing to do
            
        if just DataModule:
            create func on the fly - grab data_dim from the the DataModule
        ------------------------------------------------------------
        """
        super(scDiffEq, self).__init__(func)
        
        self.trainer = configure_lightning_trainer(model_save_dir="scDiffEq_model",
                                    log_name="lightning_logs",
                                    version=None,
                                    prefix="",
                                    flush_logs_every_n_steps=5,
                                    max_epochs=1500,
                                    log_every_n_steps=1,
                                    reload_dataloaders_every_n_epochs=5,
                                    resume_from_checkpoint=False,
                                    kwargs={},
                                   )
        
    def fit(self):
        """
        Parameters:
        -----------
        
        Returns:
        --------
        None
        
        Notes:
        ------
        """
        self.trainer.fit(self, self.DataModule())
        
    def predict(self):
        """
        Parameters:
        -----------
        
        Returns:
        --------
        None
        
        Notes:
        ------
        """
        self.pred = self.trainer.predict(self, self.DataModule())
