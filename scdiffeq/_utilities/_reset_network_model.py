import torch

def _reset_torch_model_parameters(model):
    
    """
    
    Notes:
    ------
    (1) Source: https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819
    
    """
    
    if isinstance(model, torch.nn.Conv2d) or isinstance(model, torch.nn.Linear):
        model.reset_parameters()
        
def _reset_network_model(network_model):
        
    return network_model.apply(_reset_torch_model_parameters)