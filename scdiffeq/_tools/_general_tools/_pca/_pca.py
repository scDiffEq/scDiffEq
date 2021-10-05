
from ._pca_adata import _pca_adata
from ._pca_array import _pca_array


def _pca(data):
    
    """Accepts a numpy array or AnnData."""
    
    if _is_AnnData(data):
        return _pca_adata(data)
    
    else:
        return _pca_array(data)
