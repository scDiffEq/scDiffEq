
import torch
import torchsde


def _forward_integrate(func, X, t, device="cpu", return_X=False):
    
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
    X_pred = torchsde.sdeint(func, X0, t).to(device)

    if return_X:
        return X_pred, X
    else:
        return X_pred