

def _subset_single_clonal_lineage(adata, clone_idx, silent=False):

    """"""
    single_clone_idx = adata.obs.loc[adata.obs["clone_idx"] == clone_idx].index.astype(
        int
    )
    
    clone_adata = adata[single_clone_idx]
    
    if not silent:
        print(clone_adata)
    
    return clone_adata