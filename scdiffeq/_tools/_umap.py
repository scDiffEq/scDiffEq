
__module_name__ = "_umap.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import umap


def _umap(
    adata, use_key="X_pca", key_added="X_umap", n_components=2, n_neighbors=20, silent=False, **kwargs
):

    """
    Parameters:
    -----------
    adata [ required ]
        AnnData object
        type: anndata._core.anndata.AnnData

    use_key
        key on adata.obsm_keys() by which to access an input (N,M) matrix to umap.
        default: "X_pca"
        type: str

    key_added
        key added to adata.obsm_keys() where the (N,n_components) X_umap matrix will be stored.
        default: "X_umap"
        type: str

    n_components
        [ from the umap documentation ] The dimension of the space to embed into. This defaults to 2 to
        provide easy visualization, but can reasonably be set to any integer value in the range 2 to 100.
        default: 2
        type: int

    n_neighbors
        [ from the umap documentation ] The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values result in more global views of the
        manifold, while smaller values result in more local data being preserved. In general values should
        be in the range 2 to 100.
        default: 20
        type: int
        
    silent
        If True, function does not print user messages.
        default: False
        type: bool

    Returns:
    --------
    None, adata is modified in place with:
    
    adata.obsm["X_umap"]
        Dimensionally-reduced umap embedding coordinates.
        type: numpy.ndarray
        
    adata.uns["umap"]
        A umap.UMAP() model fit to the input data using the input parameters.
        type: umap.umap_.UMAP

    kwargs:
    -------
    n_neighbors=15,
    n_components=2,
    metric='euclidean',
    metric_kwds=None,
    output_metric='euclidean',
    output_metric_kwds=None,
    n_epochs=None,
    learning_rate=1.0,
    init='spectral',
    min_dist=0.1,
    spread=1.0,
    low_memory=True,
    n_jobs=-1,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    repulsion_strength=1.0,
    negative_sample_rate=5,
    transform_queue_size=4.0,
    a=None,
    b=None,
    random_state=None,
    angular_rp_forest=False,
    target_n_neighbors=-1,
    target_metric='categorical',
    target_metric_kwds=None,
    target_weight=0.5,
    transform_seed=42,
    transform_mode='embedding',
    force_approximation_algorithm=False,
    verbose=False,
    tqdm_kwds=None,
    unique=False,
    densmap=False,
    dens_lambda=2.0,
    dens_frac=0.3,
    dens_var_shift=0.1,
    output_dens=False,
    disconnection_distance=None,
    precomputed_knn=(None, None, None),

    Notes:
    ------
    (1) Documentation for UMAP can be found here:
        https://umap-learn.readthedocs.io/en/latest/basic_usage.html
    """

    adata.uns["umap"] = umap_model = umap.UMAP(n_components=2, n_neighbors=20, **kwargs)
    adata.obsm[key_added] = umap_model.fit_transform(adata.obsm[use_key])
    
    if not silent:
        print(" - Added: adata.obsm['{}']".format(key_added))
        print(" - Added: adata.uns['umap']")
        