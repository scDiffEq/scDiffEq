import numpy as np
import torch


def _sort_data_into_batches(adata, batched_trajectory_keys):

    """
    Parameters:
    -----------
    adata

    batched_trajectory_keys

    Returns:
    --------
    Batched_Data
        type: dict
        
    Notes:
    ------

    """

    Batched_Data = {}

    for batch_n, assignment in enumerate(batched_trajectory_keys):

        Batched_Data[batch_n] = adata[
            adata.obs.loc[adata.obs.trajectory.isin(assignment)].index.astype(int)
        ]

    return Batched_Data


def _format_single_batch(batch, time_column="time_point"):

    """

    Parameters:
    -----------
    batch

    Returns:
    --------
    formatted_batch

    Notes:
    ------
    (1) Assumes that the obs table is sorted by trajectory with time increasing within each
        trajectory (i.e., t0, t1, ..., tN, t0, t1, ..., tN, ...)
    """

    t = torch.Tensor(np.sort(batch.obs[time_column].unique()))
    n_cells_batch, n_genes_batch = batch.shape[0], batch.shape[1]
    n_trajs_batch = int(n_cells_batch / len(t))
    batch_y = torch.Tensor(
        batch.X.toarray().reshape(n_trajs_batch, len(t), n_genes_batch)
    )  # N_TRAJS x N_TIMEPOINTS x N_GENES
    batch_y0 = batch_y[:, 0, :]  # N_TRAJS x N_GENES

    class _format_parallel_batch:
        def __init__(self, batch_y, batch_y0, t):

            self.batch_y = batch_y
            self.batch_y0 = batch_y0
            self.t = t

    formatted_batch = _format_parallel_batch(batch_y, batch_y0, t)

    return formatted_batch


def _format_batched_parallel_data(BatchedData, mode, time_column):

    formatted_BatchedData = {}

    for [key, batch] in BatchedData[mode].items():
        formatted_BatchedData[key] = _format_single_batch(batch, time_column)

    return formatted_BatchedData


def _format_parallel_time_batches(
    self, n_batches=20, time_column="time", verbose=False
):

    """
    Parameters:
    -----------
    adata
        AnnData

    n_batches
        default: 20
        type: int
    
    verbose
        default: False
        type: bool

    Returns:
    --------
    Batched_Data
        type: dict
        
    Notes:
    ------
    batched_data.flatten() should always be less than len(unique_trajectories)
    """

    DataObject_train = self.adata.uns["data_split_keys"]["train"]
    DataObject_valid = self.adata.uns["data_split_keys"]["validation"]

    train_unique_trajectories = DataObject_train.obs.trajectory.unique()
    valid_unique_trajectories = DataObject_valid.obs.trajectory.unique()

    BatchedData = {}
    FormattedBatchedData = {}

    batch_size_train = -1 + round(len(train_unique_trajectories) / n_batches)
    batch_size_valid = -1 + round(
        batch_size_train
        * (len(valid_unique_trajectories) / len(train_unique_trajectories))
    )

    if verbose:
        print(
            "Training data will be sorted into {} batches each consisting of {} trajectories.\n".format(
                n_batches, batch_size_train
            )
        )
        print(
            "Validation data will be sorted into {} batches each consisting of {} trajectories.\n".format(
                n_batches, batch_size_valid
            )
        )

    batched_trajectory_keys_train = np.random.choice(
        train_unique_trajectories, replace=False, size=[n_batches, batch_size_train]
    )
    batched_trajectory_keys_valid = np.random.choice(
        valid_unique_trajectories, replace=False, size=[n_batches, batch_size_valid]
    )

    BatchedData["train_batches"] = _sort_data_into_batches(
        self.adata, batched_trajectory_keys_train
    )
    BatchedData["valid_batches"] = _sort_data_into_batches(
        self.adata, batched_trajectory_keys_valid
    )

    FormattedBatchedData["train"] = _format_batched_parallel_data(
        BatchedData, time_column=time_column, mode="train_batches"
    )
    FormattedBatchedData["valid"] = _format_batched_parallel_data(
        BatchedData, time_column=time_column, mode="valid_batches"
    )

    return FormattedBatchedData
