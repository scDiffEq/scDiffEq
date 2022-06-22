
__module_name__ = "_reparameterize.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import torch


def _extract_latent_space_parameters(X_latent):

    """
    Parameters:
    -----------
    X_latent
        Encoded latent-space matrix.
        type: torch.Tensor
    Returns:
    --------
    mu, log_var
        mu: the first feature values as mean
        type: torch.Tensor
        log_var: the other feature values as variance
        type: torch.Tensor
    """
    _x = X_latent.view(-1, 2, int(X_latent.shape[-1] / 2))
    return _x[:, 0, :], _x[:, 1, :]


def _reparameterize(X_latent):

    """
    Sample from a guassian distribution in the dimensions of the input space.
    Parameters:
    -----------
    X_latent
    Returns:
    --------
    Z_sample
        Reparameterized latent vector
    Notes:
    ------
    mu
        mean from the encoder's latent space
        type: float
    log_var
        log variance from the encoder's latent space
        type: float
    """

    mu, log_var = _extract_latent_space_parameters(X_latent)

    std = torch.exp(0.5 * log_var)  # standard deviation
    eps = torch.randn_like(std)  # `randn_like` as we need the same size
    Z_sample = mu + (eps * std)  # sampling as if coming from the input space

    return Z_sample, mu, log_var