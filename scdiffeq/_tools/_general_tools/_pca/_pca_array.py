from sklearn.decomposition import PCA
from sklearn import preprocessing


def _preprocess_for_PCA(array):

    """Scale data for PCA."""

    scaler = preprocessing.StandardScaler().fit(array)
    scaled_array = scaler.transform(array)

    return scaled_array


def _pca_array(array, n_components=30, preprocess=True):

    """
    Run PCA
    
    Parameters:
    -----------
    array
        data matrix
        type: numpy.ndarray
    
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

    if preprocess:
        array = _preprocess_for_PCA(array)

    PrincipleComponents = {}

    PrincipleComponents["pca"] = pca = PCA(n_components=n_components)
    PrincipleComponents["pcs"] = pca.fit_transform(array)

    return PrincipleComponents
