
__module_name__ = "_filter_fragmented_lineages.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import numpy as np
import pandas as pd


def _filter_fragmented_lineages(
    adata, clonal_index_key="clone_idx", time_key="Time point"
):

    """
    Returns a list of [ clonal ] lineage indices with cells present at all timepoints.

    Parameters:
    -----------
    adata
        AnnData object.
        type: anndata._core.anndata.AnnData

    clonal_index_key
        pandas.DataFrame accessor
        type: str

    time_key
        pandas.DataFrame accessor
        type: str


    Returns:
    --------
    clones
        list of clones present at all timepoints
        type: numpy.ndarray

    Notes:
    ------
    Only adata.obs is required, but we pass adata for simplicity.
    """

    obs_df = adata.obs.copy()

    n_timepoints = obs_df[time_key].nunique()
    df = pd.DataFrame(
        obs_df.groupby(clonal_index_key)[time_key].nunique() >= n_timepoints
    )
    clones = np.array(df[df[time_key] == True].index).astype(int)

    return clones