
__module_name__ = "_scdiffeq.py"
__doc__ = """API-facing model module."""
__version__ = "0.0.45"
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# -- import packages: --------------------------------------------------------------------
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer
from neural_diffeqs import NeuralODE, NeuralSDE
from torch.utils.data import DataLoader
from torch_nets import TorchNet
import torch_adata
import anndata
import torch
import os


# -- import local dependencies: ----------------------------------------------------------
from . import configs


# -- API-facing model class: -------------------------------------------------------------
class scDiffEq:
    def __parse__(self, kwargs, ignore=["self"], hide=["adata", "DataModule", "func"]):

        self.PASSED_KWARGS = {}
        for key, val in kwargs.items():
            if not key in ignore:
                if key in hide:
                    setattr(self, "_{}".format(key), val)
                elif key == "kwargs":
                    for k, v in val.items():
                        self.PASSED_KWARGS[k] = v
                else:
                    self.PASSED_KWARGS[key] = val

    def __config__(self):
        self.config = configs.scDiffEqConfiguration(
            adata=self._adata, DataModule=self._DataModule, func=self._func, **self.PASSED_KWARGS
        )
        for attr in self.config.__dir__():
            if not attr.startswith("_"):
                try:
                    setattr(self, attr, getattr(self.config, attr))
                except:
                    print("Unable to set: {}".format(attr))
                    
                    
    def __init__(self,
                 adata: anndata.AnnData = None,
                 DataModule: LightningDataModule = None,
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

        self.__parse__(locals())
        self.__config__()
        
    def __repr__(self):
        return "scDiffEq model"

    def fit(self):
        self.LightningTrainer.fit(self.LightingModel, self.LightningDataModule)
