
import torch

Lin = torch.nn.functional.linear
Softmax = torch.nn.functional.softmax

def _fate_bias_transform(X_hat, n_fates, device):
    
    """
    Lin = torch.nn.Linear(X_hat.shape, n_fates)
    """

    return Softmax(Lin(X_hat, torch.ones(n_fates, X_hat.shape[1]).to(device)), dim=1)