
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
    def __parse__(self, kwargs, ignore, hide):
        
        self._SCDIFFEQ_PASSED_KWARGS = {}
        
        for key, val in kwargs.items():
            if not key in ignore:
                self._SCDIFFEQ_PASSED_KWARGS[key] = val
                    
    def __config__(self, kwargs, ignore=["self", "kwargs", "__class__", "hide"], hide=[]):

        self.__parse__(kwargs, ignore, hide)
        self.config = configs.scDiffEqConfiguration(**self._SCDIFFEQ_PASSED_KWARGS)
        for attr in self.config.__dir__():
            if (not attr.startswith("_")) and (not attr.startswith("Prediction")):
                setattr(self, attr, getattr(self.config, attr))

    def __init__(
        self,
        adata: anndata.AnnData = None,
        func: [NeuralODE, NeuralSDE, torch.nn.Module] = None,
        use_key="X_pca",
        groupby="Time point",
        obs_keys=['W', 'v'],
        train_key="train",
        val_key="val",
        test_key="test",
        predict_key="predict",
        accelerator="gpu",
        devices=1,
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
        dt=0.1,
        model_save_dir: str = "scDiffEq_model",
        log_name: str = "lightning_logs",
        version=None,
        prefix="",
        flush_logs_every_n_steps=5,
        max_epochs=1500,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=5,
        ckpt_outputs_frequency=50,
        t=None,
        **kwargs
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

        self.__config__(locals(), ignore=['self'], hide=[])
        
    def __repr__(self):
        # TODO: add a nice self-report method to be returned as a str
        return "scDiffEq model"

    def fit(self):
        self.LightningTrainer.fit(self.LightingModel, self.LightningDataModule)
        
    def test(self):
        self.test_pred = self.LightningTrainer.test(self.LightingModel, self.LightningDataModule)

    def predict(self, adata=None):
        if adata:
            self.config._reconfigure_LightningDataModule(adata)
            PredictionLightningDataModule = self.config.PredictionLightningDataModule
        else:
            PredictionLightningDataModule = self.config.LightningDataModule
        
        self.predicted = self.LightningTrainer.predict(self.LightingModel, PredictionLightningDataModule)
