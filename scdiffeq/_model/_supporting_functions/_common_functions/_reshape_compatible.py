
import torch

def _reshape_compatible(pred_y):
    
    """"""
    
    reshaped_outs = []
    for i in range(pred_y.shape[1]):
        reshaped_outs.append(pred_y[:, i, :])
        
    return torch.stack(reshaped_outs)