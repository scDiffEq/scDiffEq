
from ..utilities._preprocess import preprocess

def _pca_plot(adata, title, alpha, edgecolor, linewidths):
    
    from vintools.plotting import presets as plot_presets
    import matplotlib.pyplot as plt
    
    plot_presets(title, "PC-1", "PC-2")
    plt.scatter(
        adata.obsm['X_pca'][:, 0],
        adata.obsm['X_pca'][:, 1],
        c=adata.obs.time,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidths=linewidths,
    )
    plt.show()

def _pca(adata, n_components=2, plot=False, title=None, preprocessing=False, alpha=1, edgecolor=None, linewidths=None):
    
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    """
    Runs sklearn implementation of PCA. 
    
    Deletes any previous values stored at adata.obsm["X_pca"]. 
    """
    
    try:
        del adata.obsm["X_pca"]
    except:
        pass
    
    if preprocessing == True:
        preprocess(adata)
    
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(adata.X)
    adata.uns["pca"] = pca
    adata.obsm["X_pca"] = pcs
    
    print("Top {} PCs stored at adata.obsm['X_pca']".format(pcs.shape[1]), '\n', "PCA stored at adata.uns['pca']")

    if plot == True:
        _pca_plot(adata, title, alpha, edgecolor, linewidths)