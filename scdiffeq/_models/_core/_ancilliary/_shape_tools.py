
import torch

def _restack_x(x, t):
    return torch.stack([x[:, i, :] for i in range(len(t))])