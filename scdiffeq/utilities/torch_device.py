import torch

def set_device(gpu=None):
    
    """
    Parameters:
    -----------
    gpu
        the number (typically 0) indicating the GPU attached to the running instance.
        
    Returns:
    --------
    device
        object to which torch.Tensors can be sent to for calculations.
    
    """

    device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")

    return device

def torch_device(array, device=set_device()):
    
    """
    Parameters:
    -----------
    array
        Typically a numpy array. Object to be transformed into a torch.Tensor
    
    device
        object to which torch.Tensors can be sent to for calculations. 
        
    Returns:
    --------
    tensor
        A torch.Tensor loaded onto the specified device.
    
    """
    
    tensor = torch.Tensor(array).to(device)

    return tensor