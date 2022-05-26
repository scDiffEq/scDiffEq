
__module_name__ = "_KL_Divergence.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import torch


def _KL_Divergence(mu, logvar):

    """
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    Parameters:
    -----------
    mu
        the mean from the latent vector
    logvar
        log variance from the latent vector
        
    Return:
    -------
    KL_Divergence_loss
    """

    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())