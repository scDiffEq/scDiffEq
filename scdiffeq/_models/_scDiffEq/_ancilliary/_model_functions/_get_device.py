
__module_name__ = "_get_device.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import torch


def _get_device(device=0):

    cuda = torch.cuda.is_available()

    if cuda:
        return "cuda:{}".format(device)
    else:
        return "cpu"