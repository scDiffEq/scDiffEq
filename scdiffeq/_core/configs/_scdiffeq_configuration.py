
__module_name__ = "_configuration.py"
__doc__ = """TODO."""
__version__ = """0.0.45"""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)

from pytorch_lightning import LightningDataModule, Trainer
import torch_adata
from torch.utils.data import DataLoader


from ..lightning_model import LightningModel
from ._lightning_model_configuration import LightningModelConfig
from ._lightning_trainer_configuration import LightningTrainerConfig
from ._lightning_data_module_configuration import LightningDataModuleConfig

from ..utils import extract_func_kwargs, Base
# from ...data import LightningAnnDataModule


#     def __parse__(self, kwargs, , hide=[]):

#         self._PASSED_KWARGS = {}
#         for key, val in kwargs.items():
#             if not key in ignore:
#                 self._PASSED_KWARGS[key] = val
#                 if key in hide:
#                     key = "_{}".format(key)
#                 setattr(self, key, val)
#             elif key == "kwargs":
#                 self._PASSED_KWARGS = self._split_kwargs(val, self._PASSED_KWARGS)

#     def _split_kwargs(self, kw, KWARG_DICT={}):
#         for k, v in kw.items():
#             KWARG_DICT[k] = v
#             setattr(self, k, v)
#         return KWARG_DICT
    
class scDiffEqConfiguration(Base):
    """
    Manage the interaction with: LightningModel, Trainer, and LightningDataModule
    # called from within scDiffEq
    """
    
    def __config__(self, kwargs, ignore=["self", "kwargs"], hide=['data', 'func']):
        
        self.__parse__(locals(), ignore, hide)        
        
        self.DATA_KWARGS = extract_func_kwargs(LightningDataModuleConfig, self._PASSED_KWARGS)
        self.MODEL_KWARGS = extract_func_kwargs(LightningModel, self._PASSED_KWARGS)
        self.TRAINER_KWARGS = extract_func_kwargs(LightningTrainerConfig, self._PASSED_KWARGS)
    
    def __init__(self,
                 data=None,
                 func=None,
                 accelerator="gpu",
                 devices=1,
                 max_epochs=10,
                 log_every_n_steps=1, 
                 use_key="X_pca",
                 **kwargs):
        
        super(scDiffEqConfiguration, self).__init__()
        
        self.__config__(locals())
            
    def _configure_lightning_model(self):
        """sets self._LightningModel"""
        self.MODEL_KWARGS['state_size'] = self.state_size
        self._LightningModel = LightningModel(func=self._func, lit_config=LightningModelConfig, **self.MODEL_KWARGS)
        
    @property
    def state_size(self):
        return self.LightningDataModule.n_dim
        
    def _configure_lightning_trainer(self):
        """sets self._LightningTrainer"""
        self._LightningTrainer = LightningTrainerConfig(**self.TRAINER_KWARGS).trainer

    def _configure_lightning_data_module(self):
        """sets self._LightningDataModule"""
        self._LightningDataModule = LightningDataModuleConfig(self._data, self.DATA_KWARGS).LightningDataModule
                
#         if isinstance(self._DataModule, LightningDataModule):
#             self._LightningDataModule = self._DataModule
        
#         self._adata.obs["W"] = self._adata.obs["v"] = 1
#         dataset = torch_adata.AnnDataset(
#             self._adata, use_key="X_pca", groupby="Time point", obs_keys="W"
#         )
#         

    @property
    def LightingModel(self):
        self._configure_lightning_model()
        return self._LightningModel

    @property
    def LightningTrainer(self):
        self._configure_lightning_trainer()
        return self._LightningTrainer

    @property
    def LightningDataModule(self):
        self._configure_lightning_data_module()
        return self._LightningDataModule
