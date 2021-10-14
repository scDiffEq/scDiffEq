
# package imports #
# --------------- #
import vintools as v


def _choose_loss_function(loss_function):

    """
    Choose a loss function from torch.nn as called by a string.

    Parameters:
    -----------
    loss_function
        Loss function to be used during neural DiffEq training.
        type: str

    Returns:
    --------
    imported_torch_loss_func
        type: torch.nn.loss.<LossFunc_module>

    Notes:
    ------
    (1) Uses `v.ut.import_from_string("torch", "nn", loss_function)` to 
        flexibly return an optimizer.
    """

    imported_torch_loss_func = v.ut.import_from_string(
        package="torch", module="nn", function=loss_function
    )()

    return imported_torch_loss_func
