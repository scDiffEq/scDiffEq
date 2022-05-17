
__module_name__ = "_cuda_device.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import licorice_font as font
import torch


def _check_for_cuda_device(verbose=False):

    """
    Looks for an available cuda device.

    Parameters:
    -----------
    verbose:
        Optionally silence / recieve function feedback.
        default: False
        type: bool

    Returns:
    --------
    device
        type: str

    Notes:
    ------
    (1) Slightly friendlier function for convenience in that it first looks to see
    if cuda is available. Then looks for a device.
    """

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        if verbose:
            print(
                " - Selected Cuda device: {} available.".format(
                    font.font_format(device, ["BOLD"])
                )
            )
        device = "cuda" + str(device)
    else:
        device = "cpu"
        if verbose:
            print(
                " - Cuda is not available. Defaulting to {}.".format(
                    font.font_format(device, ["BOLD"])
                )
            )

    return device


def _assign_cuda_device(device=False, verbose=False):

    """
    Looks for an available cuda device.

    Parameters:
    -----------
    device [ optional ]
        The user can provide a cuda device as an integer or as "cpu" (type: str)
        default: False
        type: str or int

    verbose:
        Optionally silence / recieve function feedback.
        default: False
        type: bool

    Returns:
    --------
    device
        type: str

    Notes:
    ------
    (1) Slightly friendlier function for convenience in that it first looks to see
        if cuda is available. Then looks for a device. If none is found or 'cpu' is provided,
        the returned result is "cpu".
    """

    if device is "cpu":
        if verbose:
            print(
                " - Selected device: {}. \n - For large datasets, performance may be dramatically improved with the use of a GPU.".format(
                    font.font_format(device, ["BOLD"])
                )
            )

    if not device:
        if verbose:
            print(" - No device provided...")
        device = _check_for_gpu_device()

    return device