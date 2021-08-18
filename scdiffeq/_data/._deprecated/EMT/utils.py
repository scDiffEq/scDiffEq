import numpy as np
import anndata as a


def grab_stable_state_conditions(
    df, timepoint, time_label="time", include=[1, 2, 3, 4, 5]
):
    return df.loc[df[time_label] == timepoint].values[:, include][0, :]


def data_to_anndata(data, trajectories, cols, time_length):

    data["trajectory"] = trajectories
    data = data.sort_values("time").reset_index(drop=True)
    adata = a.AnnData(
        data[["miR200", "mZEB", "miR34", "mSNAIL"]]
    )  # remove first and last (time and input signal vector)
    adata.obs["time"] = data.time.values
    adata.obs["i2"] = data.i2.values
    adata.obs["trajectory"] = data.trajectory.values
    adata.uns["number_of_trajectories"] = trajectories.shape[0] / time_length

    return adata


def record_trajectory(list_of_trajectories, time_length, trajectory_number):

    # record a trajectory value for each point along the simulated trajectory. Book-keeping function.

    observed_trajectory = np.full(time_length, int(trajectory_number))
    list_of_trajectories = np.append(list_of_trajectories, observed_trajectory)

    return list_of_trajectories


def remove_nan(array):

    """Takes numpy array."""

    original = array
    array = array[~np.isnan(array).any(axis=1)]

    difference = original.shape[0] - array.shape[0]
    where = np.where(np.isnan(original).any(axis=1))[0]
    print(difference, "rows eliminated at line", where)

    return array, where
