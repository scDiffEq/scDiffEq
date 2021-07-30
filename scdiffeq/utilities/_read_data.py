def _read_h5ad(path, X_to_array=False):

    from anndata import read_h5ad

    adata = read_h5ad(path)
    
    if X_to_array:
        adata.X = adata.X.toarray()
    
    if type(adata.var.index[0]) == str:
        adata.var = adata.var.reset_index()    
        adata.var = adata.var.rename({"index": "gene_id"}, axis=1)

    print(adata)
    
    return adata