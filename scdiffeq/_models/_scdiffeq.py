
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


# -- import local dependencies: ----------------------------------------------------------
from ._base._core._base_model import LightningModel
from ._base._core._configure import configure_lightning_trainer
from ._base._core._configure import InputConfiguration
from ._base._core._base_utility_functions import extract_func_kwargs
from ._base._core._scdiffeq_datamodule import scDiffEqDataModule


# -- base model: -------------------------------------------------------------------------
class BaseModel(ABC):
    """
    Base model to interface PyTorch-Lightning model with a
    Lightning Trainer, an AnnData / LightningDataModule.
    """

    def __parse__(self, kwargs, ignore=["self", "__class__"]):
        self._kwargs = {}
        for k, v in kwargs.items():
            if not k in ignore:
                setattr(self, k, v)
                self._kwargs[k] = v

    def __configure__(self, kwargs):
        """Where we define the Trainer / loggers / etc."""
        
        self.__parse__(kwargs)
        
        lit_kwargs = extract_func_kwargs(LightningModel, self._kwargs)
        trainer_kwargs = extract_func_kwargs(configure_lightning_trainer, self._kwargs)
        data_kwargs = extract_func_kwargs(scDiffEqDataModule, self._kwargs)
        
        self.LightningModel = LightningModel(**lit_kwargs)
        self.trainer = configure_lightning_trainer(**trainer_kwargs)
        
        print("DATA KWARGS")
        print(data_kwargs)
        
        self.DataModule = scDiffEqDataModule(**data_kwargs) # percent_val=0.2, time_key=self.time_key) # 
        
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
                 optimizer_kwargs: dict = {"lr": 1e-4},
                 scheduler_kwargs: dict = {"step_size": 20, "gamma": 0.1},
                 ignore_t0: bool = True,
                 dt: float = 0.1,
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
        self.__configure__(locals())
