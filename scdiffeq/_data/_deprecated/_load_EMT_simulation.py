import vintools as v
import numpy as np
from vintools.utilities import pyGSUTILS
from .._utilities._AnnData_handlers._read_write._read_AnnData import _read_AnnData
from .._utilities._subsetting_functions import _isolate_trajectory
import glob, os

gsutil = pyGSUTILS()


def _download_EMT_simulation(
    destination_path="./scdiffeq_data", silent=False, return_data_path=False
):

    _h5ad_path = (
        "scdiffeq-data/EMT_simulation/EMT.simulation.500trajectories.AnnData.h5ad"
    )
    _pkl_path = (
        "scdiffeq-data/EMT_simulation/EMT.simulation.500trajectories.AnnData.pca.pkl"
    )
    print(
        "Downloading simulated EMT data to: {}".format(
            v.ut.format_pystring(destination_path, ["RED", "BOLD"])
        )
    )
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    path_to_data_h5ad = os.path.join(destination_path, os.path.basename(_h5ad_path))
    path_to_data_pkl = os.path.join(destination_path, os.path.basename(_pkl_path))

    if os.path.exists(path_to_data_h5ad):
        print(
            "\nData already downloaded! Using cached data from {}".format(
                v.ut.format_pystring(path_to_data_h5ad, ["BOLD", "RED"])
            )
        )
    else:
        gsutil.cp(_h5ad_path, destination_path)

    if os.path.exists(path_to_data_pkl):
        print(
            "\nData already downloaded! Using cached data from {}".format(
                v.ut.format_pystring(path_to_data_pkl, ["BOLD", "RED"])
            )
        )
    else:
        gsutil.cp(_pkl_path, destination_path)

    glob.glob(destination_path + "/*")
    if return_data_path:
        return [path_to_data_h5ad, path_to_data_pkl]


def _count_datapoints_per_trajectory(adata):

    """"""

    traj_lengths = np.array([])

    for i in adata.obs.trajectory.unique():
        traj_lengths = np.append(traj_lengths, _isolate_trajectory(adata, i).shape[0])

    mean_traj_length = traj_lengths.mean()


def _load_simulated_EMT_dataset(
    destination_path="./scdiffeq_data",
    downsample_n_trajectories=None,
    downsample_percent=1,
):

    [h5ad_path, pkl_path] = _download_EMT_simulation(
        destination_path=destination_path, silent=False, return_data_path=True
    )

    adata = _read_AnnData(
        outpath="./",
        scdiffeq_outs_dir=destination_path,
        label="EMT.simulation.500trajectories.AnnData",
        downsample_n_trajectories=downsample_n_trajectories,
        downsample_percent=downsample_percent,
    )
    adata.uns["n_datapoints_per_trajectory"] = _count_datapoints_per_trajectory(adata)

    return adata
