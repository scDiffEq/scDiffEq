import anndata

def _read_h5ad(h5ad_path, silent=False):
    
    adata = anndata.read_h5ad(h5ad_path)
    
    if not silent:
        print(adata)
    
    return adata