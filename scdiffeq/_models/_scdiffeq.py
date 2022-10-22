
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

from ._base._core._base_model import LightningModel
from ._base._core._prepare_lightning_data_module import prepare_LightningDataModule
from ._base._core._configure_lightning_trainer import configure_lightning_trainer
from ._base._core._base_utility_functions import extract_func_kwargs
# -- import local dependencies: ----------------------------------------------------------
# from . import _base as base
from pytorch_lightning import Trainer


# base.prepare_LightningDataModule(adata)

# -- import packages: --------------------------------------------------------------------
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base model to interface PyTorch-Lightning model with a
    Lightning Trainer, an AnnData / LightningDataModule.
    """

    def __parse__(self, kwargs, ignore=["self", "__class__"]):
        ref_kwargs = {}
        for k, v in kwargs.items():
            if not k in ignore:
                setattr(self, k, v)
                ref_kwargs[k] = v
        self.kwargs = ref_kwargs

    def __configure__(self):
        """Where we define the Trainer / loggers / etc."""
        
        lit_kwargs = extract_func_kwargs(LightningModel, self.kwargs)
        trainer_kwargs = extract_func_kwargs(configure_lightning_trainer, self.kwargs)
        data_kwargs = extract_func_kwargs(prepare_LightningDataModule, self.kwargs)
        
        self.trainer = Trainer(accelerator="gpu", devices=1, max_epochs=20)
        self.LightningModel = LightningModel(**lit_kwargs)
        self.DataModule = prepare_LightningDataModule(**data_kwargs)
        # configure_lightning_trainer(**trainer_kwargs)

    def fit(self):
        self.trainer.fit(self.LightningModel, self.DataModule)

    def predict(self):
        self.pred = self.trainer.predict(self, self.DataModule)

        
## == moved code: ==== was in base model. more effective here        
#     def __register_inputs__(self):
#         """To-do: docs"""
#         pass

#         self.config = InputConfiguration(func=self.func, adata=self.adata, DataModule=self.DataModule)
#         self.config.configure(use_key=self.use_key, time_key=self.time_key, w_key=self.w_key, **kwargs)
#         self.config.pass_to_model(self)
        
#         self.func = func
#         self.dt = dt
#         self.save_hyperparameters(ignore=["func"])
#         if hasattr(self, "func"):
#             self.__configure_forward_step__(self.ignore_t0)
#
## ============================================================

# -- Focus of this module: scDiffEq model: -----------------------------------------------
class scDiffEq(BaseModel):
    
    def __init__(self,
                 adata: anndata.AnnData = None,
                 DataModule: LightningDataModule=None,
                 func:[NeuralODE, NeuralSDE, torch.nn.Module] = None,
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
        
        super(scDiffEq, self)
        self.__parse__(locals())
        
#         self.trainer = base.configure_lightning_trainer(model_save_dir="scDiffEq_model",
#                                     log_name="lightning_logs",
#                                     version=None,
#                                     prefix="",
#                                     flush_logs_every_n_steps=5,
#                                     max_epochs=1500,
#                                     log_every_n_steps=1,
#                                     reload_dataloaders_every_n_epochs=5,
#                                     kwargs={},
#                                    )
