
import anndata

def idx_to_int_str(adata: anndata.AnnData, drop: bool = False) -> anndata.AnnData:
    """
    """
    adata.obs = adata.obs.reset_index(drop=drop)
    adata.obs.index = adata.obs.index.astype(str)
    return adata