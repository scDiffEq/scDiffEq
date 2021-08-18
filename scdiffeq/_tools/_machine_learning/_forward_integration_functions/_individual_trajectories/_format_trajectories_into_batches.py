# package imports #
# --------------- #
import numpy as np
import torch


def _check_overlap_bewteen_data_subsets(adata, subsets=["train", "validation", "test"]):

    """
    This function checks between data subsets within AnnData to see if there is any overlap between groups.
    
    Parameters:
    -----------
    adata
        AnnData
        
    subsets
        default: ['train', 'validation', 'test']
        type: list of str
        
    Returns:
    --------
    prints n_overlap. Desired: 0
    
    Notes:
    ------
    """

    df = adata.obs

    overlaps = []
    print("Checking between...\n")
    for subset in subsets:
        for subset_check_against in subsets:
            if subset != subset_check_against:
                print("\t", subset, subset_check_against)
                overlaps.append(
                    df.loc[df[subset] == True]
                    .loc[df[subset_check_against] == True]
                    .shape[0]
                )
    n_overlap = np.array(overlaps).sum()


def _get_formatted_trajectory(data_object, trajectory):

    """"""

    obs_df = data_object.obs
    traj_idx = obs_df.loc[obs_df.trajectory == trajectory].index.astype(int)

    traj_data = data_object.data[traj_idx]
    traj_df = obs_df.iloc[traj_idx]
    traj_t = data_object.t.iloc[traj_idx]
    try:
        traj_emb = data_object.emb[traj_idx]
    except:
        traj_emb = None

    class _formatted_trajectory:
        def __init__(self, index, data, obs, t, emb):

            self.index = index
            self.data = data
            self.y = torch.Tensor(data)
            self.y0 = torch.Tensor(data[0])
            self.obs = obs
            self.t = torch.Tensor(t.values)
            self.t0 = self.t
            self.emb = emb

    formatted_trajectory = _formatted_trajectory(
        index=traj_idx, data=traj_data, obs=traj_df, t=traj_t, emb=traj_emb
    )
    return formatted_trajectory


def _format_trajectories(data_object):

    """
    Parameters:
    -----------
    data_object

    Returns:
    --------

    Notes:
    ------
    (1)  `data_object` is not to be confused with the AnnData object.

    """

    formatted_trajectories = {}

    unique_trajectories = data_object.obs.trajectory.unique()
    for trajectory in unique_trajectories:
        formatted_trajectories[trajectory] = _get_formatted_trajectory(
            data_object, trajectory
        )

    return formatted_trajectories


def _Trajectory_DataDict_to_batches(formatted_trajs, batch_assignments):

    """
    formatted_trajs is a dict
    """

    BatchDict = {}

    for batch_n, assignment in enumerate(batch_assignments):
        batch = []
        for traj in assignment:
            batch.append(formatted_trajs[traj])
        BatchDict[batch_n] = batch

    return BatchDict


def _split_formatted_trajectories_into_batches(formatted_trajs, n_batches):

    """
    formatted_trajs

    n_batches

    Notes:
    ------
    (1) pass one subset at a time (i.e., train or val) as `formatted_trajs`
    """

    batch_size = round(len(formatted_trajs) / n_batches)
    batch_assignments = np.random.choice(
        list(formatted_trajs.keys()), [n_batches, batch_size], replace=False
    )

    batched_data = _Trajectory_DataDict_to_batches(formatted_trajs, batch_assignments)

    return batched_data


def _get_formatted_training_validation_trajectories(adata, n_batches):

    train = adata.uns["data_split_keys"]["train"]
    valid = adata.uns["data_split_keys"]["validation"]

    train_trajs = _format_trajectories(train)
    valid_trajs = _format_trajectories(valid)

    BatchedData = {}

    BatchedData["train_batches"] = _split_formatted_trajectories_into_batches(
        train_trajs, n_batches
    )
    BatchedData["valid_batches"] = _split_formatted_trajectories_into_batches(
        valid_trajs, n_batches
    )

    return BatchedData
