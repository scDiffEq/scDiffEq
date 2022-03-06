import licorice
import os
import pydk
import torch

def _save_torch_model(model, outpath):
    
    state = model.state_dict()    
    torch.save(state, outpath)
    
    return outpath