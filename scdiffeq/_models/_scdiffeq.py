
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
from abc import ABC, abstractmethod
from pytorch_lightning import LightningDataModule
from neural_diffeqs import NeuralODE, NeuralSDE
from torch_composer import TorchNet
import anndata
import torch
import os


# -- import local dependencies: ----------------------------------------------------------
from ._base._core._base_model import LightningModel
from ._base._core._configure import configure_lightning_trainer
from ._base._core._configure import InputConfiguration
from ._base._core._base_utility_functions import extract_func_kwargs
from ._base._core._scdiffeq_datamodule import configure_data


# -- base model: -------------------------------------------------------------------------
class BaseModel(ABC):
    """
    Base model to interface PyTorch-Lightning model with a
    Lightning Trainer, an AnnData / LightningDataModule.
    """

    def __parse__(self, kwargs, ignore):
        self._kwargs = {}
        for k, v in kwargs.items():
            if k == "kwargs":
                for l, w in v.items():
                    self._kwargs[l] = w
            elif not k in ignore:
                setattr(self, k, v)
                self._kwargs[k] = v

    def __report_kwargs__(self, lit_kwargs, trainer_kwargs, data_kwargs, ignore):
        
        ignore += ["func", "adata"]
        
        print("\n - LIGHTNING MODEL KWARGS -")
        print("----------------------------")
        for k, v in lit_kwargs.items():
            if not k in ignore:
                print("{}: {}".format(k, v))
        
        print("\n - DATA KWARGS -")
        print("-----------------")
        for k, v in trainer_kwargs.items():
            if not k in ignore:
                print("{}: {}".format(k, v))
        
        print("\n - TRAINER / LOGGER KWARGS -")
        print("----------------------------")
        for k, v in data_kwargs.items():
            if not k in ignore:
                print("{}: {}".format(k, v))
        
                
    def __configure__(self, report_kwargs, kwargs, ignore=["self", "__class__"]):
        """Where we define the Trainer / loggers / etc."""
        
        self.__parse__(kwargs, ignore)
        
        lit_kwargs = extract_func_kwargs(LightningModel, self._kwargs)
        trainer_kwargs = extract_func_kwargs(configure_lightning_trainer, self._kwargs)
        data_kwargs = extract_func_kwargs(configure_data, self._kwargs)
        
        if report_kwargs:
            self.__report_kwargs__(lit_kwargs, trainer_kwargs, data_kwargs, ignore)
        
        self.LightningModel = LightningModel(**lit_kwargs)
        self.trainer = configure_lightning_trainer(**trainer_kwargs)
        self.DataModule = configure_data(**data_kwargs)
        
    def fit(self):
        self.trainer.fit(self.LightningModel, self.DataModule)

    def test(self):
        self.test_pred = self.trainer.test(self, self.DataModule)
        
    def predict(self):
        self.pred = self.trainer.predict(self, self.DataModule)


# -- Focus of this module: scDiffEq model: -----------------------------------------------
class scDiffEq(BaseModel):
    
    def __init__(self,
                 adata: anndata.AnnData = None,
                 DataModule: LightningDataModule=None,
                 func:[NeuralODE, NeuralSDE, torch.nn.Module] = None,
                 time_key="Time point",
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
        
        super(scDiffEq, self)
        self.__configure__(report_kwargs, locals())
