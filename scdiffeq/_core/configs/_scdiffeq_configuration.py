
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

from ..utils import extract_func_kwargs

class scDiffEqConfiguration:
    """
    Manage the interaction with: LightningModel, Trainer, and LightningDataModule
    # called from within scDiffEq
    """
    
    def __parse__(self, kwargs, ignore=['self'], hide=['adata', 'DataModule', 'func']):
        
        self.PASSED_KWARGS = {}
        
        for key, val in kwargs.items():
            if not key in ignore:
                if key in hide:
                    key = "_{}".format(key)
                elif key == "kwargs":
                    for _key, _val in val.items():
                        self.PASSED_KWARGS[_key] = _val
                else:
                    self.PASSED_KWARGS[key] = val
                setattr(self, key, val)
        
        # TODO: add data config func so you can pass the following:
        # self.DATA_KWARGS = extract_func_kwargs(LightningData, self.PASSED_KWARGS)
        
        self.MODEL_KWARGS = extract_func_kwargs(LightningModel, self.PASSED_KWARGS)
        self.TRAINER_KWARGS = extract_func_kwargs(Trainer, self.PASSED_KWARGS)
    
    def __init__(self,
                 adata=None,
                 DataModule=None,
                 func=None,
                 accelerator="gpu",
                 devices=1,
                 max_epochs=10,
                 log_every_n_steps=1, 
                 **kwargs):
        
        self.__parse__(locals())
            
    def _configure_lightning_model(self):
        """sets self._LightningModel"""
        self._LightningModel = LightningModel(func=self._func, lit_config=LightningModelConfig, **self.MODEL_KWARGS)

    def _configure_lightning_trainer(self):
        """sets self._LightningTrainer"""
        self._LightningTrainer = Trainer(**self.TRAINER_KWARGS)

    def _configure_lightning_data_module(self):
        """sets self._LightningDataModule"""
                
        if isinstance(self._DataModule, LightningDataModule):
            self._LightningDataModule = self._DataModule
        
#         self._adata.obs["W"] = self._adata.obs["v"] = 1
#         dataset = torch_adata.AnnDataset(
#             self._adata, use_key="X_pca", groupby="Time point", obs_keys="W"
#         )
#         self._LightningDataModule = DataLoader(dataset, batch_size=5200, num_workers=4)

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
