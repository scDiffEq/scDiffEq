import torch

def load_model(model, ckpt_path, device):
    
    """useful when you have a live pytorch-lightning model"""
    
    state = torch.load(ckpt_path, map_location=device)
    epoch = state["epoch"]
    model.load_state_dict(state["state_dict"])

    return model, epoch