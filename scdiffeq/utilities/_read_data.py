def _read_h5ad(path):

    from anndata import read_h5ad

    adata = read_h5ad(path)

    print(adata)
    
    return adata