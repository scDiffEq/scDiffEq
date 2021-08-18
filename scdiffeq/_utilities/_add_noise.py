import numpy as np


def _add_noise(adata, noise=0.05, return_adata=False):

    """
    Adds random noise to adata.X. Loops through each gene and adds noise relative to
    the mean expression value of that gene. In my experience, I get the best results 
    when preprocessing is run *before* and *after* running this function. The first 
    preprocessing scales the data such that the noise added to each gene is equally 
    meaningful. The second round of preprocessing is performed such that values with 
    noise are scaled.
    
    Parameters:
    ----------
    adata
        AnnData object. 

    noise
        default: 0.05
            
    Returns:
    --------
    
    None
        adata is modified in place.
    """

    adata.X = adata.X + np.random.normal(0, noise, adata.X.shape)

    if return_adata:
        return adata
