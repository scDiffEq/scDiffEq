
from ._pca_adata import _pca_adata
from ._pca_array import _pca_array

def _is_AnnData(data_object):

    """Determine if data object is AnnData."""

    return data_object.__class__.__name__ == "AnnData"

def _pca(data, n_components=30, preprocess=False, plot=False):
    
    """
    Run PCA. Accepts a numpy array or AnnData.
    
    Parameters:
    -----------
    data
        data matrix (adata.X for AnnData)
        type: numpy.ndarray or AnnData
    
    n_components [optional]
        number of dimensions 
        default: 30
        type: int
        
    preprocess
        calls `sklearn.preprocessing.StandardScaler` function from sklearn that removes mean and scales 
        by unit variance. 
        default: True
        type: boolean
    
    Notes:
    ------
    (1) Function used as implemented by sklearn.
        (a) https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        
        (b) From the source: "Linear dimensionality reduction using Singular Value Decomposition 
            of the data to project it to a lower dimensional space. The input data is centered 
            but not scaled for each feature before applying the SVD.
    """
    
    if _is_AnnData(data):
        return _pca_adata(data, plot, title="PCA", preprocessing=preprocess, alpha=1, edgecolor=None, linewidths=None,)
    
    else:
        return _pca_array(data, n_components, preprocess)
