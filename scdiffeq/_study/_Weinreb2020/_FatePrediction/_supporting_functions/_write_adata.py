
import pickle

def _write_adata(adata, path):

    """"""

    adata_ = adata.copy()    

    umap_savepath = path.strip(".h5ad") + ".umap_adata"
    pca_savepath = path.strip(".h5ad") + ".pca_adata"
    
    pickle.dump(adata_.uns["pca"], open(pca_savepath, 'wb'))
    pickle.dump(adata_.uns["umap"], open(umap_savepath, 'wb'))
    
    for transformer in ['scaler', 'pca', 'umap']:
        del adata_.uns[transformer]

    adata_.write_h5ad(path)