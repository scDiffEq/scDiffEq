from ...._utilities._preprocess import preprocess


def _pca_adata(
    adata,
    plot=False,
    title=None,
    preprocessing=False,
    alpha=1,
    edgecolor=None,
    linewidths=None,
):

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    """
    Runs sklearn implementation of PCA. 
    
    Deletes any previous values stored at adata.obsm["X_pca"]. 
    Incomplete as of right now. 
    """

    try:
        del adata.obsm["X_pca"]
    except:
        pass

    if preprocessing == True:
        preprocess(adata)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(adata.X)
    adata.uns["pca"] = pca
    adata.obsm["X_pca"] = pcs

    if plot == True:

        plt.scatter(
            pcs[:, 0],
            pcs[:, 1],
            c=adata.obs.time,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidths=linewidths,
        )
