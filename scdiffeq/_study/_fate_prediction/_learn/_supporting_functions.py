import numpy as np
import os
import torch
import pandas as pd

def _get_unique_ordered(series, reverse=False):

    """ """

    if reverse:
        return np.sort(series.unique())[::-1]
    else:
        return np.sort(series.unique())


def _sample_cell_indices_within_clonal_lineage_for_training(
    obs_df, idx, clonal_index_key="clone_idx", time_key="Time point"
):

    """Return a dictionary corresponding to sampled cell indices where each timepoint = len(n_cells_t_final)"""

    grouped_by_clones = obs_df.groupby([clonal_index_key, time_key])
    t_all = _get_unique_ordered(df[t_key])

    clone_t_final = grouped_by_clones.get_group((idx, t_all[-1])).index.astype(int)
    n_samples = len(clone_t_final)

    CloneIdx = {}
    for t in t_all[:-1]:
        CloneIdx[t] = (
            grouped_by_clones.get_group((idx, t))
            .sample(n_samples, replace=True)
            .index.astype(int)
        )
    CloneIdx[t_all[-1]] = clone_t_final

    return CloneIdx


def _make_batch_list(shuffled_indices, batch_bounds):

    batches = []

    for i in range(len(batch_bounds)):
        if batch_bounds[i] == np.max(batch_bounds):
            batches.append(shuffled_indices[batch_bounds[i] :])
        else:
            batches.append(shuffled_indices[batch_bounds[i] : batch_bounds[i + 1]])

    return batches


def _make_epoch_batches(epoch_data, batch_size):

    """Shuffle batch indices and set up bounds in which those indices will be batched

    Return a list of these batches
    """

    n_lineages = epoch_data.shape[1]
    shuffled_indices = np.random.choice(n_lineages, n_lineages)
    batch_bounds = np.arange(start=0, stop=len(shuffled_indices), step=batch_size)

    return _make_batch_list(shuffled_indices, batch_bounds)


def _setup_logfile(outdir, columns=["epoch", "d2", "d4", "d6", "total"]):

    status_file = open(os.path.join(outdir, "status.log"), "w")
    header = "\t".join(columns) + "\n"
    status_file.write(header)
    status_file.flush()

    return status_file


def _update_logfile(status_file, epoch, epoch_loss):

    """"""

    epoch_loss = epoch_loss.sum(axis=0).detach().numpy().tolist()
    epoch_loss_ = [epoch] + epoch_loss + [sum(epoch_loss)]
    epoch_loss_ = [str(loss) for loss in epoch_loss_]
    status_update = "\t".join(epoch_loss_) + "\n"
    status_file.write(status_update)
    status_file.flush()


def _return_clones_present_at_all_timepoints(
    adata, clonal_index_key="clone_idx", time_key="Time point"
):

    """Returns a list lineages with data at ALL timepoints."""

    n_timepoints = adata.obs[time_key].nunique()

#     return np.argwhere(
#         adata.obs.groupby(clonal_index_key)[time_key].nunique().values >= n_timepoints
#     ).flatten()

    df = pd.DataFrame(adata.obs.groupby(clonal_index_key)[time_key].nunique() >= n_timepoints)
    clones = np.array(df[df[time_key] == True].index).astype(int)
    
    return clones

def _sample_clonal_lineages(
    X_pca, obs_df, clones, clonal_idx_key="clone_idx", time_key="Time point"
):

    """
    Takes < 1s for sampling the entire LARRY dataset


    To time:
    --------
    t0 = time.time()
    for i in range(20):
        epoch_data = _sample_clonal_lineages(adata)
    tf = time.time()
    """

    grouped_by_clones = obs_df.loc[obs_df[clonal_idx_key].isin(clones)].groupby(
        [clonal_idx_key, time_key]
    )
    t_all = _get_unique_ordered(obs_df[time_key])

    all_clones_sampled = []

    for clone in clones:
        clone_t_final = grouped_by_clones.get_group((clone, t_all[-1])).index.astype(
            int
        )
        n_samples = len(clone_t_final)
        clone_idxs = np.zeros([3, n_samples])
        clone_idxs[-1] = clone_t_final
        for i in range(len(t_all) - 1):
            clone_idxs[i] = (
                grouped_by_clones.get_group((clone, t_all[i]))
                .sample(n_samples, replace=True)
                .index.astype(int)
            )
        all_clones_sampled.append(clone_idxs)

    clone_idx_sampling = np.hstack(all_clones_sampled).astype(int)  # t x n_cells at d6

    return torch.Tensor(X_pca[clone_idx_sampling.astype(int)])


def _report_loss(epoch, epoch_loss):
    
    epoch_loss_by_day = "  ".join(
        epoch_loss.sum(0).detach().numpy().astype(str).tolist()
    )
    message = "Epoch {} | Wasserstein Training Loss | {} | Sum: {:.2f}".format(
        epoch, epoch_loss_by_day, epoch_loss.sum().item()
    )
    print(message)
