
__module_name__ = "_scdiffeq_configuration.py"
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
from torch.utils.data import DataLoader
import torch_adata
import anndata
import torch

from ._lightning_model_configuration import LightningModelConfig
from ._lightning_trainer_configuration import LightningTrainerConfiguration
from ._lightning_data_module_configuration import LightningDataModuleConfig

from ..utils import function_kwargs, AutoParseBase

from ._configure_time import TimeConfig
from autodevice import AutoDevice
import logging


logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


class scDiffEqConfiguration(AutoParseBase):
    """
    Manage the interaction with: LightningModel, Trainer, and LightningDataModule
    # called from within scDiffEq
    """
    
    CONFIG_KWARGS = {}
    
    def __init__(self, **kwargs):
        super(scDiffEqConfiguration, self).__init__()
        self.__config__(locals())

    def __config__(self, kwargs, ignore=["self", "__class__", "hide"]):

        
        self.__parse__(kwargs['kwargs'], ignore=ignore)

        self.config_t = TimeConfig(**function_kwargs(TimeConfig, self._KWARGS))
        time_kwargs = self.config_t()
        self._KWARGS.update(time_kwargs)
        for k, v in time_kwargs.items():
            setattr(self, k, v)

        self.CONFIG_KWARGS["MODEL"] = function_kwargs(
            LightningModelConfig, self._KWARGS
        )
        self.CONFIG_KWARGS["TRAINER"] = function_kwargs(
            LightningTrainerConfiguration, self._KWARGS
        )
        self.CONFIG_KWARGS["DATA"] = function_kwargs(
            LightningDataModuleConfig, self._KWARGS
        )

    # -- key properties: ------------------------------------------------------------
    @property
    def LightningModel(self):
        return LightningModelConfig(**self.CONFIG_KWARGS["MODEL"]).LightningModel
    
    @property
    def LightningDataModule(self):
        return LightningDataModuleConfig(**self.CONFIG_KWARGS["DATA"]).LightningDataModule

#     @property
#     def LightningTrainer(self):
#         return LightningTrainerConfig(**self.KWARGS["TRAINER"]).trainer
    
#     @property
#     def LightningTestTrainer(self):
#         return LightningTrainerConfig(**self.KWARGS["TRAINER"]).test_trainer

#     def _reconfigure_LightningDataModule(self, adata, N=False):
#         self.KWARGS["DATA"]['adata'] = adata
#         self.KWARGS["DATA"]['N'] = N
#         self._PredictionLightningDataModule = LightningDataModuleConfig(**self.KWARGS["DATA"]).LightningDataModule
#         return self._PredictionLightningDataModule
        
#     @property
#     def PredictionLightningDataModule(self):
#         return self._PredictionLightningDataModule
