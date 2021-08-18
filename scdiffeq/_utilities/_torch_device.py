import torch


def _set_device(gpu="0"):

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


def _torch_device(array, device=_set_device()):

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
