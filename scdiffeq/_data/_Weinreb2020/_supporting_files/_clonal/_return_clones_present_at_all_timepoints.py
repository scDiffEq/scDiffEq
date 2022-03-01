import numpy as np
import pandas as pd

from ._count_clones import _count_clones

def _return_clones_present_at_all_timepoints(
    adata, 
    clonal_index_key="clone_idx", 
    time_key="Time point", 
    plot=True,
    silent=False
):

    """
    Returns a list lineages with data at all timepoints in the dataset.

    Parameters:
    -----------
    adata
        AnnData
        type: AnnData.anndata

    clonal_index_key
        column of adata.obs containing annotation of clonal lineage to which the cell belongs.
        type: str
        default: "clone_idx"

    time_key
        column of adata.obs containing time annotation.
        type: str
        default: "Time point"
    
    silent
        If True, messages are not returned. 
        type: bool
        default: False
        
    Returns:
    --------
    [adata_filt, counted_clones_df, filtered_clonal_idx]
        type: list
    
        adata_filt
            AnnData object filtered for only clones present in all 3 timepoints.
            type: anndata.AnnData
            
        counted_clones_df
            pandas.DataFrame with two columns: [["clone", "count"]]
            type: pandas.DataFrame
            
        filtered_clonal_idx
            Array of clones present at all timepoints.
            type: numpy.ndarray
    
    """

    n_timepoints = adata.obs[time_key].nunique()
    obs_grouped_by_clone = adata.obs.groupby(clonal_index_key)
    n_clones = len(obs_grouped_by_clone)
    df = pd.DataFrame(obs_grouped_by_clone[time_key].nunique() >= n_timepoints)
    filtered_clonal_idx = np.array(df[df[time_key] == True].index).astype(int)
    perc_clones_at_all_t = (len(filtered_clonal_idx) / n_clones) * 100
    
    if not silent:
        print(
            "{}/{} ({:.2f}%) clones present at all time points.".format(
                len(filtered_clonal_idx), n_clones, perc_clones_at_all_t
            )
        )
        
    adata_filt = adata[adata.obs.loc[adata.obs[clonal_index_key].isin(filtered_clonal_idx)].index.astype(int)]
    counted_clones_df = _count_clones(adata, adata_filt, clonal_index_key, plot)
    
    return [adata_filt, counted_clones_df, filtered_clonal_idx]