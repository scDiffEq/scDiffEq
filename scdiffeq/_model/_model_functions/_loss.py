import pandas as pd
import numpy as np
import torch

from .._model_utilities._log import _update_logfile

def _release_from_memory(EpochLoss):

    """
    Function to release loss values from GPU memory.

    Parameters:
    -----------
    EpochLoss : dict
        dict of epoch loss tensors
        tensors are of variable shape.

    Returns:
    --------
    EpochLoss : dict
        dict of epoch loss tensors
        tensors are of variable shape.
    """

    for time_format, data_group in EpochLoss.items():
        for _key, _tensor in data_group.items():
            EpochLoss[time_format][_key] = _tensor.detach()
    return EpochLoss


def _sum_backprop_loss(EpochLoss, optimizer):

    """
    Sum and backproporgation of loss values.

    Parameters:
    -----------
    EpochLoss : dict
        dict of epoch loss tensors
        tensors are of variable shape.

    optimizer : torch.optim
        torch optimizer

    Returns:
    --------
    EpochLoss : dict
        dict of epoch loss tensors
        tensors are of variable shape.

    """
    optimizer.zero_grad()

    total_loss_ = []
    for time_group in EpochLoss.keys():
        time_group_loss = EpochLoss[time_group]["train_loss"].sum()
        total_loss_.append(time_group_loss)
    reported_loss = torch.stack(total_loss_).sum()
    reported_loss.backward()
    optimizer.step()
    reported_loss = reported_loss.detach()

    return _release_from_memory(EpochLoss)


def _loop_through_lineage_loss(X_pred, X, X_lineage_idx, LossFunc):

    lineage_loss_ = []
    start, end = X_lineage_idx[:-1], X_lineage_idx[1:]

    for i, j in zip(start, end):
        x_pred_lineage = X_pred[:, i:j]
        x_lineage = X[:, i:j]
        lineage_loss = LossFunc(x_pred_lineage.contiguous(), x_lineage.contiguous())
        lineage_loss_.append(lineage_loss)

    return lineage_loss_


def _linage_aware_loss(X_pred, X, X_lineage_idx, LossFunc, test):

    if test:
        with torch.no_grad():
            return torch.stack(
                _loop_through_lineage_loss(X_pred, X, X_lineage_idx, LossFunc)
            )

    else:
        return torch.stack(
            _loop_through_lineage_loss(X_pred, X, X_lineage_idx, LossFunc)
        )


def _calculate_loss(X_pred_, optimizer, LossFunc, test=False):

    EpochLoss = {}

    for time_group in X_pred_.keys():
        EpochLoss[time_group] = {}
        EpochLoss[time_group]["train_loss"] = _linage_aware_loss(
            X_pred_[time_group]["X_pred"],
            X_pred_[time_group]["X"],
            X_pred_[time_group]["X_train_idx"],
            LossFunc,
            test,
        )
        if test:
            if "X_test" in X_pred_[time_group].keys():
                EpochLoss[time_group]["test_loss"] = _linage_aware_loss(
                    X_pred_[time_group]["X_pred_test"],
                    X_pred_[time_group]["X_test"],
                    X_pred_[time_group]["X_test_idx"],
                    LossFunc,
                    test,
                )

    return _sum_backprop_loss(EpochLoss, optimizer)

def _create_loss_summary_report_dict(EpochLoss, mean_loss=True, sum_loss=False):

    """
    Reformat loss dict of tensor outputs from each cell.

    This will turn the values of EpochLoss into a sorted dataframe.

    Parameters
    ----------
    EpochLoss : dict
        dict of EpochLoss.
        the format is {time_format: {'train_loss': tensor,
        'test_loss': tensor}}

    mean_loss : bool, optional
        whether to take the mean of the loss per cell, by default True

    sum_loss : bool, optional
        whether to take the sum of the loss per cell, by default False

    Returns
    -------
    dict
        dict with loss values for each cell. will be of the form
        {str : np.ndarray}

    Notes:
    ------
    (1) Can do more but for now, the focus is on the time-resolved loss.

    """

    LossReport = {}

    for time_group, grouped_dict in EpochLoss.items():
        LossReport[time_group] = {}
        for test_train, loss_tensor in grouped_dict.items():
            LossReport[time_group][test_train] = {}

            if mean_loss:
                LossReport[time_group][test_train]["mean_t_loss"] = loss_tensor.mean(0)
                LossReport[time_group][test_train][
                    "mean_lineage_loss"
                ] = loss_tensor.mean(1)
            if sum_loss:
                LossReport[time_group][test_train]["sum_t_loss"] = loss_tensor.sum(0)
                LossReport[time_group][test_train][
                    "sum_lineage_loss"
                ] = loss_tensor.sum(1)

    return LossReport


def _get_subset_loss_df(LossReport, subset="train", loss_type="mean_t_loss"):

    SubsetLossDict = {}
    for dt_type in LossReport.keys():
        for data_split in LossReport[dt_type].keys():
            if subset in data_split:
                for loss in LossReport[dt_type][data_split].keys():
                    if loss == loss_type:

                        SubsetLossDict[dt_type] = {}
                        values = LossReport[dt_type][data_split][loss]
                        t = np.array(dt_type.split("_")).astype(float).astype(int)
                        for _t, _loss in zip(t, values):
                            SubsetLossDict[dt_type][_t] = _loss.item()
    return pd.DataFrame.from_dict(SubsetLossDict)


def _get_summarized_loss(LossReport, loss_type="mean_t_loss"):

    train_df = _get_subset_loss_df(LossReport, subset="train", loss_type="mean_t_loss")
    test_df = _get_subset_loss_df(LossReport, subset="test", loss_type="mean_t_loss")
    summarized_loss_df = pd.concat([train_df.mean(1), test_df.mean(1)], axis=1)
    summarized_loss_df.columns = ["train", "test"]

    return summarized_loss_df.dropna(axis=1)

    
def _echo_loss(loss_df, status_file, epoch, silent=False):

    """"""

    cols = loss_df.columns.tolist()
    loss_dict = loss_df.to_dict()

    for subset in cols:
        if not silent:
            print("{} - {:<5} - |".format(epoch, subset.upper()), end=" ")
        for k, v in loss_dict[subset].items():
            if not silent:
                print("{}: {:.3f}".format(k, v), end="\t")
        print("Total: {:.3f}".format(sum(loss_dict[subset].values())))
        _update_logfile(
            status_file, epoch, epoch_loss=loss_df[subset].tolist(), mode=subset
        )

def _loss(X_pred, optimizer, loss_function, test, epoch, status_file, silent=False):

    """"""

    EpochLoss = _calculate_loss(X_pred, optimizer, loss_function, test)
    LossReport = _create_loss_summary_report_dict(
        EpochLoss, mean_loss=True, sum_loss=False
    )
    summarized_loss_df = _get_summarized_loss(LossReport, loss_type="mean_t_loss")
    _echo_loss(summarized_loss_df, status_file, epoch, silent)

    return summarized_loss_df