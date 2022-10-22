
import torch_adata
from torch_adata import BaseLightningDataModule
from pytorch_lightning import LightningDataModule
import inspect
import torch
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import os
import anndata
from abc import ABC, abstractmethod
from ._base_utility_functions import extract_func_kwargs

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
    use_key="X_pca",
    w_key="W",
    time_key="Time point",
    percent_val=0.2,
    train_key="train",
    test_key="test",
)->LightningDataModule:

    return DataModule(
        adata=adata,
        use_key=use_key,
        w_key=w_key,
        time_key=time_key,
        percent_val=percent_val,
        train_key=train_key,
        test_key=test_key,
    )