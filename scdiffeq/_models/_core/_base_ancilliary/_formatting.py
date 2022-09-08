
import torch

# ------------------------------------------------------------------------------------- #

def _restack(x, t):
    return torch.stack([x[:, i] for i in range(len(t))])


def _format_batched_inputs(batch, t):

    """
    re-orders with time as the first dimension.
    """
    
    X, W, F = batch
    
    X0 = X[:, 0, :]
    X_obs = _restack(X, t)
    W_obs = _restack(W, t)

    return X0, X_obs, W_obs, F

# ------------------------------------------------------------------------------------- #
