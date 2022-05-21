import numpy as np


def _filter_lineages_by_timepoint_occupance(
    adata, min_timepoints=2, time_key="Time point", lineage_key="clone_idx"
):
    """
    Parameters:
    -----------
    adata
        AnnData object.
        type: anndata._core.anndata.AnnData
        
    min_timepoints
        default: 2
        type: int
        
    time_key
        default: "Time point"
        type: str
        
    lineage_key
        default: "clone_idx"
        type: str

    Returns:
    --------
    adata_filtered
        AnnData object.
        type: anndata._core.anndata.AnnData
    
    Notes:
    ------
    (1)
    """

    df = adata.obs.copy()

    passing_lineages = []

    for clone, clone_df in df.groupby(lineage_key):
        unique_t = np.sort(clone_df[time_key].unique())
        if len(unique_t) >= min_timepoints:
            passing_lineages.append(clone)

    return adata[df.loc[df[lineage_key].isin(passing_lineages)].index]