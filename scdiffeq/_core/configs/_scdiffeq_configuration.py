
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
from ._lightning_trainer_configuration import LightningTrainerConfig
from ._lightning_data_module_configuration import LightningDataModuleConfig

from ..utils import function_kwargs, Base

from ._configure_time import TimeConfig

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


class scDiffEqConfiguration(Base):
    """
    Manage the interaction with: LightningModel, Trainer, and LightningDataModule
    # called from within scDiffEq
    """
    KWARGS = {}
    def __config__(self, kwargs, ignore=["self", "kwargs", "__class__", "hide"]):

        self.__parse__(kwargs, ignore)
        self.config_t = TimeConfig(**function_kwargs(TimeConfig, self._KWARGS))
        time_kwargs = self.config_t()
        self._KWARGS.update(time_kwargs)
        for k, v in time_kwargs.items():
            setattr(self, k, v)

        self.KWARGS["MODEL"] = function_kwargs(
            LightningModelConfig, self._KWARGS
        )
        self.KWARGS["TRAINER"] = function_kwargs(
            LightningTrainerConfig, self._KWARGS
        )
        self.KWARGS["DATA"] = function_kwargs(
            LightningDataModuleConfig, self._KWARGS
        )

    def __init__(
        self,
        adata: anndata.AnnData = None,
        func=None,
        use_key="X_pca",
        time_key=None,
        t0_idx=None,
        dt=0.1,
        n_steps=40,
        obs_keys=None,
        t_min=0,
        t_max=1,
        t=None,
        train_key="train",
        val_key="val",
        test_key="test",
        predict_key="predict",
        accelerator="gpu",
        devices=1,
        adjoint=False,
        batch_size=2000,
        num_workers=4,
        n_groups=None,
        train_val_percentages=[0.8, 0.2],
        remainder_idx=-1,
        predict_all=True,
        attr_names={"obs": [], "aux": []},
        one_hot=False,
        aux_keys=None,
        silent=True,
        optimizer="RMSprop",
        lr_scheduler="StepLR",
        lr=0.0001,
        step_size=20,
        model_save_dir: str = "scDiffEq_model",
        log_name: str = "lightning_logs",
        version=None,
        prefix="",
        flush_logs_every_n_steps=5,
        max_epochs=1500,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=5,
        ckpt_outputs_frequency=50,
        N=False,
        **kwargs
    ):
        
        super(scDiffEqConfiguration, self).__init__()
        self.__config__(locals())

    # -- key properties: ------------------------------------------------------------
    @property
    def LightingModel(self):
        return LightningModelConfig(**self.KWARGS["MODEL"]).LightningModel

    @property
    def LightningTrainer(self):
        return LightningTrainerConfig(**self.KWARGS["TRAINER"]).trainer
    
    @property
    def LightningTestTrainer(self):
        return LightningTrainerConfig(**self.KWARGS["TRAINER"]).test_trainer

    @property
    def LightningDataModule(self):
        return LightningDataModuleConfig(**self.KWARGS["DATA"]).LightningDataModule

    def _reconfigure_LightningDataModule(self, adata, N=False):
        self.KWARGS["DATA"]['adata'] = adata
        self.KWARGS["DATA"]['N'] = N
        self._PredictionLightningDataModule = LightningDataModuleConfig(**self.KWARGS["DATA"]).LightningDataModule
        return self._PredictionLightningDataModule
        
    @property
    def PredictionLightningDataModule(self):
        return self._PredictionLightningDataModule
