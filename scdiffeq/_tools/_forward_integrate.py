
import torch
import torchsde


def _forward_integrate(func, X, t, device="cpu"):
    
    """
    
    Parameters:
    -----------
    X
    
    t
    
    func
    
    device
    
    Returns:
    --------
    X_pred, X
    
    Notes:
    ------
    """

    X0 = X[0].to(device)
    t = torch.sort(t)[0].to(device)
    func.batch_size = X0.shape[0]
    X_pred = torchsde.sdeint(func, X0, t).to(device) # , dt_min=0.01

    return X_pred, X