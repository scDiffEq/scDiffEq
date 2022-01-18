
import anndata as a
import pickle

def _read_inputs(adata_path, umap_path):

    """"""
    
    adata = a.read_h5ad(adata_path)
    umap = pickle.load(open(umap_path, "rb"))

    return adata, umap