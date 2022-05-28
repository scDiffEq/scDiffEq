
__module_name__ = "_utilities.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import numpy as np
import torch


def _get_device(device=0):

    cuda = torch.cuda.is_available()

    if cuda:
        return "cuda:{}".format(device)
    else:
        return "cpu"
    
    
def _transfer_attributes(fetch, transfer_to):
    
    """
    utility function to transfer attributes of one class to the a new class. Useful if 
    you do not want to pass all attributes as arguments to the inhereting class. Might be
    eliminated with the use of super() but I'm not sure how to do that at the moment.
    """

    keys = np.array(fetch.__dir__())[[not i.startswith("__") for i in fetch.__dir__()]]
    for key in keys:
        transfer_to.__setattr__(key, fetch.__getattribute__(key))

    return transfer_to