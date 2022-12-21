
__module_name__ = "_auto_device.py"
__doc__ = """To-do"""
__author__ = "Michael E. Vinyard"
__email__ = "mvinyard@broadinstitute.org"


# -- import packages: --------------------------------------------------------------------
import torch


# -- import / define types: --------------------------------------------------------------
NoneType = type(None)
from typing import Union


# -- Functional class object: ------------------------------------------------------------
class _AutoDevice:
    def __init__(self, use_cpu=False, idx=None, *args, **kwargs):

        self._use_cpu = use_cpu
        self._idx = idx

        if "cpu" in args:
            self._use_cpu

    def availability(self):

        self.has_cuda = torch.cuda.is_available()
        self.has_mps = torch.backends.mps.is_available()
        if not any([self.has_cuda, self.has_mps]):
            self._use_cpu = True

    def configure(self):
        self.availability()

        if self.has_mps:
            self.device = torch.device("mps")
        elif self.has_cuda:
            if not self._idx:
                self._idx = torch.cuda.current_device()
            self.device = torch.device("cuda:{}".format(self._idx))

        else:
            self.device = torch.device("cpu")

    def __call__(self, *args, **kwargs):
        self.configure(*args)

        if not self._use_cpu:
            return self.device
        else:
            return self.device


# -- API-facing function: ----------------------------------------------------------------
def auto_device(
    use_cpu: bool = False, idx: Union[NoneType, int] = None, *args, **kwargs
):
    """
    Parameters:
    -----------
    use_cpu
        type: bool
        default: False
    
    idx
        type: Union[NoneType, int]
        default: None
    
    Returns:
    --------
    device
        type:
            torch.device
    """

    device = _AutoDevice(use_cpu=use_cpu, idx=idx, *args, **kwargs)
    return device()
