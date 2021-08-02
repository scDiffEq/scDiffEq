
import numpy as np

def _add_noise(adata, noise=0.05, uniform=False, return_adata=False):

    """
    Adds random noise to adata.X. Loops through each gene and adds noise relative 
    to the mean expression value of that gene. This function should be run before 
    preprocessing such that values with noise are scaled.
    
    Parameters:
    ----------
    adata
        AnnData object. 
    
    
    noise
        default: 0.05
    
    uniform
        assumes all genes are the same scale. proceeds more quickly. 
        
    Returns:
    --------
    
    None
        adata is modified in place.
    """
    
    if uniform:
        adata.X = adata.X + np.random.normal(0, noise, adata.X.shape)
        
    else:
        mean_gex_vals = adata.X.mean(axis=0)
        noisy_data_X = np.zeros(adata.X.shape)

        for gene, gene_mean in enumerate(mean_gex_vals):
            single_gene_data = adata.X[:, gene]
            noisy_data_X[:, gene] = single_gene_data*noise + np.random.normal(
                0, gene_mean, single_gene_data.shape
            )
    
    if return_adata:
        return adata