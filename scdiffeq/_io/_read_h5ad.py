
from anndata import read_h5ad

def _read_h5ad(h5ad_path, silent=False):
    
    adata = read_h5ad(h5ad_path)
    
    if not silent:
        print(adata)
    
    return adata