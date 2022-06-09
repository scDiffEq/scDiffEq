
import numpy as np

def _mk_X_augment_zeros(X_data, augment_dims):
    return np.zeros([X_data.shape[0], augment_dims])

def _augment_X_input(adata, use_key, X_data, augment_dims):
    
    X_zeros = _mk_X_augment_zeros(X_data, augment_dims)
    
    adata.obsm[use_key] = np.hstack([adata.obsm[use_key], np.zeros([adata.shape[0], augment_dims])])
    return adata, np.hstack([X_data, X_zeros])