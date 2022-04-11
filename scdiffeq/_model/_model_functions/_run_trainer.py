
__module_name__ = "_scDiffEq_Model.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pydk
import time
import torch

from ..._tools._OptimalTransportLoss import _OptimalTransportLoss

loss_func = _OptimalTransportLoss(0)

def _count_previous_epochs(LossTracker):
    
    count = 0
    for loss_list in LossTracker.values():
        count += len(loss_list)
        
    return count

def _min_max_normalize_time(t):
    return (t - t.min()) / (t.max() - t.min())


def _get_time(lineage_df, time_key="Time point", normalize=True):

    """
    df
        pandas.DataFrame containing a column denoting time via `time_key`

    time_key
        df column accessor for time
        default: "Time point"
        type: str

    normalize
        default: True
        type: bool
    """

    t = lineage_df.sort_values(time_key)[time_key].unique()
    t_norm = _min_max_normalize_time(t)

    if normalize:
        time_df = pd.DataFrame([t, t_norm], index=[time_key, "t_norm"]).T
        return t_norm, lineage_df.merge(time_df, on=time_key)

    else:
        return t, lineage_df


def _retrieve_data(X_data, lineage_df, cell_idx_key="cell_idx"):

    """"""

    t_norm, lineage_df = _get_time(lineage_df)
    n_final = lineage_df.loc[lineage_df["t_norm"] == t_norm.max()].shape[0]
    groupby_t = lineage_df.groupby("t_norm")
    len_t = len(t_norm)

    lineage_idx, X_lineage = [], []

    for n, (t, t_df) in enumerate(groupby_t):
        if t < t_norm.max():
            _lineage_idx = np.random.choice(
                t_df[cell_idx_key].to_numpy().astype(int), n_final
            )
        else:
            _lineage_idx = t_df[cell_idx_key].to_numpy().astype(int)
        lineage_idx.append(_lineage_idx)
        X_lineage.append(torch.Tensor(X_data[_lineage_idx]))

    return torch.stack(X_lineage), lineage_idx  # t x n x dim


def _get_lineage_y0(X_lineage_data):
    return torch.concat([lineage[0, :, :] for lineage in X_lineage_data])


def _prepare_lineage_data(adata, X_data, notebook):

    """"""

    X_lineage_data, X_lineage_indices = [], []
    grouped_by_lineage = adata.obs.groupby("clone_idx")

    for lineage, lineage_df in grouped_by_lineage: # tqdm(grouped_by_lineage):
        X_lineage, lineage_idx = _retrieve_data(X_data, lineage_df)
        X_lineage_data.append(X_lineage)
        X_lineage_indices.append(lineage_idx)

    return X_lineage_data, _get_lineage_y0(X_lineage_data), X_lineage_indices

def _calculate_loss(X_data, X_lineage_indices, pred, loss_func, optimizer):

    """calculate loss by lineage"""

    LossDict = {}
    evaluated = 0

    for n, lineage in enumerate(X_lineage_indices):
        X_true = torch.stack([torch.Tensor(X_data[t]) for t in lineage])
        n_cell_int = X_true.shape[1]
        X_pred = pred[:, evaluated : evaluated + n_cell_int, :]
        evaluated = int(evaluated + n_cell_int)
        LossDict[n] = loss_func(X_true.contiguous(), X_pred.contiguous())

    return LossDict


def _run_single_epoch(
    epoch,
    previous_epochs,
    X_data,
    X_idx,
    X0,
    nn_func,
    forward_integrate,
    optimizer,
    t,
    status_file,
    time_it=False,
):

    """
    Key function

    Parameters:
    -----------

    Returns:
    --------

    Notes:
    ------
    Step 1: run forward integration (hopefully all-together)
    Step 2: Get loss (might have to do in batches).
    """
    
    epoch = epoch + previous_epochs

    start_int = time.time()
    nn_func.batch_size = X0.shape[0]
    X_pred = forward_integrate(nn_func.to(0), X0.to(0), t.to(0)).to(0)
    end_int = time.time()

    loss_calc_start = time.time()
    LossDict = _calculate_loss(X_data, X_idx, X_pred, loss_func, optimizer)
    loss = torch.stack(list(LossDict.values()))
    to_print = loss.mean(0).tolist()
    
    loss_update = "Epoch: {}\t| d2: {:.1f}\t| d4: {:.2f}\t| d6: {:.2f}\t | Total: {:.2f}".format(epoch,
                                                                                                 to_print[0],
                                                                                                 to_print[1],
                                                                                                 to_print[2],
                                                                                                 sum(to_print))
    _update_logfile(status_file, epoch=epoch, epoch_loss=to_print)
    
    if epoch % 10 == 0:
        print(loss_update)    
        
    loss_calc_end = time.time()
    total_loss = loss.sum()
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if time_it:
        print("Forward integration: {} s".format(end_int - start_int))
        print("Loss calculation: {} s".format(loss_calc_end - loss_calc_start))
    
    reported_loss = sum(to_print)
    
    return X_pred, reported_loss

def _run_trainer(run_outdir,
                 epochs,
                 adata,
                 X_data, 
                 nn_func,
                 int_func,
                 optimizer,
                 LossTracker,
                 status_file,
                 epoch_counter,
                 t=torch.Tensor([0, 0.01, 0.02]),
                 time_it=False,
                 notebook=True):

    """"""
    
    if notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm    
        
    current_run = list(LossTracker.keys())[-1]
    loss_tracking = LossTracker[current_run] # list
    
    if not current_run == 0:
        previous_epochs = _count_previous_epochs(LossTracker)
    else:
        previous_epochs = 0
        
    complete_lineages = _filter_fragmented_lineages(adata)
    adata_LineageTraced = adata[adata.obs["clone_idx"].isin(complete_lineages)]
    X_lineage_data, X0, X_idx = _prepare_lineage_data(adata_LineageTraced, X_data, notebook)
    
    pydk.mkdir_flex("{}/img".format(run_outdir))
    pydk.mkdir_flex("{}/model".format(run_outdir))
    
    for epoch in tqdm(range(epochs)):
        X_pred, X_loss = _run_single_epoch(
            epoch_counter, previous_epochs, X_data, X_idx, X0, nn_func, int_func, optimizer, t, status_file, time_it
        )
        loss_tracking.append(X_loss)
        epoch_counter += 1
        if (previous_epochs + epoch_counter % 10) == 0:
            torch.save(nn_func.state_dict(), "{}/model/{}_epochs.model".format(run_outdir, epoch))
        