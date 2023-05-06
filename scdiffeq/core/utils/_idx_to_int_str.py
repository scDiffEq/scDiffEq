
import anndata

def idx_to_int_str(adata: anndata.AnnData) -> anndata.AnnData:
    """
    """
    adata.obs = adata.obs.reset_index()
    adata.obs.index = adata.obs.index.astype(str)
    return adata