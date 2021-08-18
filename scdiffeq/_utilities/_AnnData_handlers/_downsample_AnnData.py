import numpy as np
from ._format_AnnData import _format_AnnData_mtx_as_numpy_array


def _downsample_AnnData(
    adata, percent=1, n_traj=False, sort_on=["trajectory", "time"], silence=False
):

    """
    Downsamples AnnData in (4) steps:
    ---------------------------------
    (1) Ensure AnnData.X is formatted as an array
    (2) [optional] subset by number of trajectories
    (3) sorts AnnData on some parameter; this should be used (if desired) to make sure Anndata is sampled evenly. 
    (4) samples AnnData evenly based on input percent.
    
    Parameters:
    -----------
    adata
        AnnData object.

    percent
        experiment-specific label.
    
    n_traj
        number of trajectories to include
        type: int
       
    sort_on
        df keys contained in adata.obs on which AnnData object should be sorted prior to downsampling
        default: ["trajectory", "time"] (useful for scdiffeq; might eventually change to make more general or if implemented in vintools.)

    silence
        scdiffeq-specific outs directory

    Returns:
    --------
    None
    
    """

    # (1) make sure adata.X is a numpy array
    _format_AnnData_mtx_as_numpy_array(adata)

    # (2) if desired, subset trajectories from the full dataset
    if n_traj:
        trajectory_subset = np.random.choice(
            adata.obs.trajectory.unique(), n_traj
        ).tolist()
        adata = adata[
            adata.obs.loc[adata.obs.trajectory.isin(trajectory_subset)].index.astype(
                int
            )
        ]
        adata.obs = adata.obs.reset_index(drop=True)

    # (3) if desired, sort adata to make sure you sample correctly (i.e., across trajectories, time, etc. )
    sorted_adata = adata[adata.obs.sort_values(sort_on).index.astype(int)]
    sorted_adata.obs = sorted_adata.obs.reset_index(drop=True)

    # (4) downsample, inverting the desired percent retained to take every nth (i.e., sample_factor) item
    sample_factor = int(percent ** -1)
    sampled_sorted_adata = adata[::sample_factor]

    # reset index
    sampled_sorted_adata.obs = sampled_sorted_adata.obs.reset_index(drop=True)

    if not silence:
        print(sampled_sorted_adata)

    return sampled_sorted_adata
