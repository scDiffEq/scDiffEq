
"""
Flexible fetch of optimizer, LR_scheduler, other functions.

"""

# -- import packages: ----------------------------------------------------------
import torch

# -- import local dependencies: ------------------------------------------------
import ABCParse


# -- define types: -------------------------------------------------------------
from typing import Union, Any
NoneType = type(None)


class FunctionFetch(ABCParse.ABCParse):
    """Fetch a function from a flexible input."""
    def __init__(self, module=None, parent=None):
        self.__parse__(locals())

    def __call__(self, func: Union[Any, str]):
        if isinstance(func, str):
            return getattr(self.module, func)
        elif issubclass(func, self.parent):
            return func
        else:
            print("pass something else....")
            
# -- supporting functions: -----------------------------------------------------
def fetch_optimizer(func):
    """
    Examples:
    ---------
    fetch_optimizer("RMSprop")
    >>> torch.optim.rmsprop.RMSprop

    fetch_optimizer(torch.optim.RMSprop)
    >>> torch.optim.rmsprop.RMSprop
    """
    fetch = FunctionFetch(module=torch.optim, parent=torch.optim.Optimizer)
    return fetch(func)


def fetch_lr_scheduler(func):
    """
    Examples:
    ---------
    fetch_lr_scheduler("StepLR")
    >>> torch.optim.lr_scheduler.StepLR

    fetch_lr_scheduler(torch.optim.lr_scheduler.StepLR)
    >>> torch.optim.lr_scheduler.StepLR
    """
    fetch = FunctionFetch(
        module=torch.optim.lr_scheduler, parent=torch.optim.lr_scheduler.LRScheduler
    )
    return fetch(func)