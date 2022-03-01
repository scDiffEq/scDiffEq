import licorice
from sklearn.cluster import KMeans

# need to finish (and plotting)

def _kmeans(
    adata,
    use_key="X_pca",
    key_added="kmeans",
    n_clusters=100,
    plot=True,
    silent=False,
    **kwargs
):

    kmeans = KMeans(n_clusters=n_clusters)
    if not silent:
        print("{}...".format(kmeans), end=" ")
    obs_key_added = "{}.k={}".format(key_added, str(n_clusters))
    adata.obs[obs_key_added] = kmeans.fit_predict(adata.obsm[use_key])
    adata.uns['kmeans_key'] = obs_key_added
    if not silent:
        print(
            "Done. Resulting cluster assignment stored as: adata.obs['{}']".format(
                licorice.font_format(obs_key_added, ["BOLD"])
            )
        )
    
    if plot:
        _plot_categorical_adata_umap(adata, key=obs_key_added)