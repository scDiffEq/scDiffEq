from typing import Union
import numpy, torch


def sum_normalize(X: Union[torch.Tensor, numpy.ndarray], sample_axis=1):
    """
    Parameters:
    -----------
    X
        torch.Tensor or nump.array

    sample_axis
        default: 1
        type: int

    Returns:
    --------
    return X_
    """
    return X / X.sum(sample_axis)[:, None]