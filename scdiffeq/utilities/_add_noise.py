
import numpy as np

def _add_noise(adata, noise=0.05):

    """
    Adds random noise to adata.X
    
    Paramters:
    ----------
    adata
    
    
    noise
        default: 0.05
    
    
    Returns:
    --------
    
    None
        adata
    """
    
    adata.X = adata.X + np.random.normal(0, noise, adata.X.shape)
    
    return adata