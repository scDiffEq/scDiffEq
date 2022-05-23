
__module_name__ = "_pca.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
from sklearn.decomposition import PCA
from sklearn import preprocessing


def _preprocess_for_pca(X, **kwargs):

    """
    Scale data for input to PCA by standarizing features using the
    sklearn.preprcessing.StandardScaler module. This function removes the
    mean of the data across the samples axis and scales to the unit
    variance of the sample. This is a typical preprocessing step prior to PCA.

    Parameters:
    -----------
    X
        type: numpy.ndarray

    kwargs:
    -------
    copy
        defalt: True
        type: bool

    with_mean
        defalt: True
        type: bool

    with_std
        defalt: True
        type: bool

    Returns:
    --------
    X_scaled
        type: numpy.ndarray

    Notes:
    ------
    (1) This function is a wrapper of sklearn.preprcessing.StandardScaler. Further
        documentation can be found here:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    """

    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)


def _ensure_numpy_array(X):
    """
    checks the class name of the passed array and converts to numpy if not numpy.
    this may actually be a pretty brittle function as the `.toarray()` method may
    not work on all array types beyond scipy.sparse. To be tested.
    """
    if X.__class__.__name__ == "ndarray":
        return X
    else:
        return X.toarray()


def _pca(adata, n_components=50, preprocess=True, key_added="X_pca", silent=False):

    """
    PCA-transform adata.X

    Parameters:
    -----------
    adata
        AnnData object.
        type: anndata._core.anndata.AnnData

    n_components
        Number of principle components in dimensionally-reduced output.
        default: 50
        type: int

    preprocess
        Indicates if the adata.X matrix should be preprocessed using sklearn.preprocessing.StandardScaler
        default: True
        type: bool
    
    silent
        If True, function does not print user messages.
        default: False
        type: bool
        
    Returns:
    --------
    None, adata is modified in place with:

        adata.obsm["X_pca"]
            pca-transformed matrix
            shape: [M, n_pcs].
            type: numpy.ndarray

        adata.uns["pca"]
            PCA model object
            type: sklearn.decomposition._pca.PCA
            
        adata.layers["scaled"]
            preprocessed, scaled input matrix
            type: numpy.ndarray

    Notes:
    ------
    (1) Documentation for sklearn.decomposition.PCA can be found here:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """

    # ------------------------------------- #
    # preprocess
    # ------------------------------------- #

    X_ = _ensure_numpy_array(adata.X)

    if preprocess:
        adata.layers["scaled"] = X_scaled = _preprocess_for_pca(X_)
        if not silent:
            print(" - Added: adata.layers[{}]".format("scaled"))
        
    else:
        X_scaled = X_

    # ------------------------------------- #
    # PCA transform
    # ------------------------------------- #

    adata.uns["pca"] = pca = PCA(n_components=n_components)
    adata.obsm[key_added] = pca.fit_transform(X_scaled)

    if not silent:
        print(" - Added: adata.obsm['{}']".format(key_added))
        print(" - Added: adata.uns['pca']")
        