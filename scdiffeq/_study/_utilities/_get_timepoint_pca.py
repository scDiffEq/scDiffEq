def _get_timepoint_pca(adata, clone_adata, t):

    """"""

    X_pca = {}
    for timepoint in t:
        t_idx = clone_adata.obs.loc[
            clone_adata.obs["Time point"] == timepoint
        ].index.astype(int)
        X_pca[timepoint] = adata.obsm["X_pca"][t_idx]

    return X_pca