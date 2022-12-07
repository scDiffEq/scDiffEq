
__module_name__ = "_configure_lightning_trainer.py"
__doc__ = """To-do."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


from abc import ABC, abstractmethod
import os

import torch_adata
from pytorch_lightning import LightningDataModule
import inspect
import torch

from torch.utils.data import DataLoader
import anndata
from abc import ABC, abstractmethod
# from ._base_utility_functions import extract_func_kwargs
from .._utils import (
#     autodevice,
    extract_func_kwargs,
    local_arg_parser,
)
from neural_diffeqs import NeuralSDE
from torch.utils.data import DataLoader


class BaseLightningDataModule(ABC, LightningDataModule):
    def __init__(
        self, adata: anndata.AnnData = None, batch_size=2000, num_workers=os.cpu_count(), **kwargs
    ):
        super().__init__()

        self.adata = adata
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
        self.__configure__()
    
    @abstractmethod
    def __configure__(self):
        pass
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
    
    
def _augment_obs_with_W(adata, w_key="W"):
    w_hat_key = "{}_hat".format(w_key)
    if not w_key in adata.obs.columns.tolist():
        adata.obs[w_key] = 1
        adata.obs[w_hat_key] = 1
    return [w_key, w_hat_key]


def _split_test_train(
    adata,
    stage,
    use_key,
    time_key,
    w_keys,
    percent_val=0.2,
    train_key="train",
    test_key="test",
):

    if all([key in adata.obs.columns.tolist() for key in [train_key, test_key]]):
        train_adata, test_adata = (
            adata[adata.obs[train_key]],
            adata[adata.obs[test_key]],
        )
        test_dataset = torch_adata.AnnDataset(
            test_adata,
            use_key=use_key,
            groupby=time_key,
            obs_keys=w_keys,
            silent=True,
        )

    else:
        train_adata, test_dataset = adata, None

    train_dataset = torch_adata.AnnDataset(
        train_adata, use_key=use_key, groupby=time_key, obs_keys=w_keys, silent=True
    )

    if percent_val:
        train_dataset, val_dataset = torch_adata.split(
            train_dataset, percentages=[(1 - percent_val), percent_val]
        )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }


class DataModule(BaseLightningDataModule):

    def __configure__(self):
        
        kw = extract_func_kwargs(torch_adata.AnnDataset, self.kwargs)
        dataset = torch_adata.AnnDataset(self.adata[::100], groupby=self.kwargs['time_key'], silent=True, **kw)
        setattr(self, "n_dims", dataset.X.shape[-1])
    
    def prepare_data(self):
        """Download, prepare, etc. only called on 1 GPU/TPU in distributed"""

        kw = extract_func_kwargs(_augment_obs_with_W, self.kwargs)
        self.w_keys = _augment_obs_with_W(self.adata, **kw)

    def setup(self, stage):
        """Assign data to train/val/test groups here. called on every process in DDP"""
        kw = extract_func_kwargs(_split_test_train, self.kwargs)
        datasets = _split_test_train(self.adata, stage, w_keys=self.w_keys, **kw)

        for k, v in datasets.items():
            if not v is None:
                setattr(self, k, v)
                
def prepare_LightningDataModule(
    adata,
    batch_size=20,
    use_key="X_pca",
    w_key="W",
    time_key="Time point",
    percent_val=0.2,
    train_key="train",
    test_key="test",
)->LightningDataModule:

    return DataModule(
        adata=adata,
        batch_siz=batch_size,
        use_key=use_key,
        w_key=w_key,
        time_key=time_key,
        percent_val=percent_val,
        train_key=train_key,
        test_key=test_key,
    )



# ----
# ----
# ----

class InputConfiguration:
    def __init__(self, func=None, adata=None, DataModule=None):
        self.__parse__(locals())

    def __parse__(self, kwargs, return_dict=False, ignore=["self"]):

        parsed_kwargs = {}
        non_null = []
        for k, v in kwargs.items():
            if not k in ignore:
                setattr(self, k, v)
                parsed_kwargs[k] = v
                if not v is None:
                    non_null.append(k)

        setattr(self, "_attrs", parsed_kwargs.keys())
        setattr(self, "_non_null", non_null)
        if return_dict:
            return parsed_kwargs

    def _adata_only(self, use_key, time_key, w_key):

        self.DataModule = prepare_LightningDataModule(
            self.adata, use_key=use_key, time_key=time_key, w_key=w_key
        )
        self.func = NeuralSDE(state_size=self.DataModule.n_dims)

    def configure(self, use_key="X_pca", time_key="Time point", w_key="w", **kwargs):
        if self._non_null == ["adata"]:
            self._adata_only(use_key, time_key, w_key)

        elif self._non_null == ["DataModule"]:
            SDE_kwargs = extract_func_kwargs(NeuralSDE, kwargs)
            self.func = NeuralSDE(state_size=self.DataModule.n_dims, **SDE_kwargs)

        elif self._non_null == ["func"]:
            print(" - [ NOTE ] | PASS ADATA OR NEURAL DIFFEQ")

        elif self._non_null == ["func", "adata"]:
            self.DataModule = prepare_LightningDataModule(
                self.adata, use_key=use_key, time_key=time_key, w_key=w_key
            )

        elif self._non_null == ["adata", "DataModule"]:
            self.func = NeuralSDE(state_size=self.DataModule.n_dims, **SDE_kwargs)

        elif self._non_null == ["func", "DataModule"]:
            pass
        elif self._non_null == ["func", "adata", "DataModule"]:
            pass

    def pass_to_model(self, model):

        for attr in self._attrs:
            setattr(model, attr, getattr(self, attr))


# -- : ----
from pytorch_lightning import loggers
from pytorch_lightning import Trainer
import os
import torch


# from ._base_utility_functions import extract_func_kwargs


def configure_CSVLogger(
    model_save_dir: str = "scDiffEq_model",
    log_name: str = "lightning_logs",
    version=None,
    prefix="",
    flush_logs_every_n_steps=5,
):
    """
    model_save_dir
    log_name
    version
    prefix
    flush_logs_every_n_steps

    Notes:
    ------
    (1) Used as the default logger because it is the least complex and most predictable.
    (2) This function simply handle the args to pytorch_lighting.loggers.CSVLogger. While
        not functionally necessary, helps to clean up model code a bit.
    (3) doesn't change file names rather the logger name and what's added to the front of
        each log name event within the model.
    (4) Versioning contained / created automatically within by lightning logger
    """

    model_log_path = os.path.join(model_save_dir, log_name)

    logger = loggers.CSVLogger(
        save_dir=model_save_dir,
        name=log_name,
        version=version,
        prefix=prefix,
        flush_logs_every_n_steps=flush_logs_every_n_steps,
    )

    return {"logger": logger, "log_path": model_log_path}


def accelerator():
    if torch.cuda.is_available():
        return "gpu"
    return "cpu"


def configure_lightning_trainer(
    model_save_dir="scDiffEq_model",
    log_name="lightning_logs",
    devices=torch.cuda.device_count(),
    version=None,
    prefix="",
    flush_logs_every_n_steps=5,
    max_epochs=1500,
    log_every_n_steps=1,
    reload_dataloaders_every_n_epochs=5,
    kwargs={},
):

    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    logger_kwargs = extract_func_kwargs(func=configure_CSVLogger, kwargs=locals())
    logging = configure_CSVLogger(**logger_kwargs)
    
    trainer_kwargs = extract_func_kwargs(func=Trainer, kwargs=locals())
    
    return Trainer(
        accelerator=accelerator(),
        logger=logging["logger"],
        **trainer_kwargs
    )
