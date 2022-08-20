

import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch_adata import TimeResolvedAnnDataset


def _prepare_LARRY_dataset(
    adata,
    train_perc=0.9,
    num_workers=os.cpu_count(),
    train_batch_size=200,
    val_batch_size=200,
    test_batch_size=200,
):

    """returns the LARRY dataset prepared for the FATE PREDICTION task."""
    
    adata.X = adata.X.toarray()

    complete_dataset = TimeResolvedAnnDataset(adata)

    train_dataset = TimeResolvedAnnDataset(adata[adata.obs["train"]])
    test_dataset = TimeResolvedAnnDataset(adata[adata.obs["test"]])

    n_train_ = train_dataset.__len__()

    n_train = int(n_train_ * train_perc)
    n_val = int(n_train_ - n_train)
    train, val = random_split(train_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train, num_workers=num_workers, batch_size=train_batch_size
    )
    val_loader = DataLoader(val, num_workers=num_workers, batch_size=val_batch_size)
    test_loader = DataLoader(
        test_dataset, num_workers=num_workers, batch_size=test_batch_size
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "complete": complete_dataset,
    }

from .._io._read_h5ad import _read_h5ad
from cell_fate import pp

import numpy as np
import pandas as pd

###
# None of this is very clean but eventually, we will clean it up. Right now, it serves
# development needs.
###

def _annotate_timepoint_recovery_train_test(adata):
    
    test, train = np.zeros(len(adata)), np.zeros(len(adata))
    test_idx = np.where(adata.obs['Time point'].isin([2,  4]))[0]
    train_idx = np.where(adata.obs['Time point'].isin([2, 6]))[0]
    test[test_idx], train[train_idx] = 1, 1

    # we could make these pd.Categorical(), which is the AnnData convention,
    # but not doing so allows us to slice the adata object directly. 
    
    adata.obs['train'] = train.astype(bool)
    adata.obs['test'] = test.astype(bool)
    
from licorice_font import font_format

def _lazy_LARRY(
    h5ad_path,
    task="fate_prediction",
    train_perc=0.9,
    num_workers=os.cpu_count(),
    train_batch_size=2000,
    val_batch_size=1000,
    test_batch_size=1000,
):
    
    """
    task:
        options: [ "fate_prediction", "timepoint_recovery" ]
    """
    
    adata = _read_h5ad(h5ad_path)
    
    
    if task == "fate_prediction":
        pp.annotate_train_test(adata)
        pp.annotate_unique_test_train_lineages(adata)
        adata.obs.index = adata.obs.index.astype(str)
    elif task == "timepoint_recovery":
        _annotate_timepoint_recovery_train_test(adata)
    else:
        opt1 = font_format("fate_prediction", ["BOLD", "BLUE"])
        opt2 = font_format("timepoint_recovery", ["BOLD", "BLUE"])
        print("Choose: {} or {}".format(opt1, opt2))
              
    dataset = _prepare_LARRY_dataset(
        adata,
        train_perc,
        num_workers,
        train_batch_size,
        val_batch_size,
        test_batch_size,
    )
    dataset["task"] = task

    return adata, dataset