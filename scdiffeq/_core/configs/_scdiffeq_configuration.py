
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

import pytorch_lightning
import torch_adata
from torch.utils.data import DataLoader


from ..lightning_model import LightningModel
from ._lightning_model_configuration import LightningModelConfig

class scDiffEqConfiguration:
    """
    Manage the interaction with: LightningModel, Trainer, and LightningDataModule
    # called from within scDiffEq
    """
    def __init__(self, adata=None, data=None, func=None):
        
        self._adata = adata
        self._data  = data
        self._func  = func
            
    def _configure_lightning_model(self):
        """sets self._LightningModel"""
        self._LightningModel = LightningModel(func=self._func, lit_config=LightningModelConfig)

    def _configure_lightning_trainer(self):
        """sets self._LightningTrainer"""
        self._LightningTrainer = pytorch_lightning.Trainer(
            accelerator="gpu", devices=1, max_epochs=10, log_every_n_steps=1
        )

    def _configure_lightning_data_module(self):
        """sets self._LightningDataModule"""
                
        if isinstance(self._data, pytorch_lightning.LightningDataModule):
            self._LightningDataModule = self._data
        
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
