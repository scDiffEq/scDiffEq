# package imports #
# --------------- #
import vintools as v


def _choose_loss_function(loss_function):

    """
    Parameters:
    -----------
    loss_function
        type: str

    Returns:
    --------
    imported_torch_loss_func
        type: torch.nn.loss.<LossFunc_module>

    Notes:
    ------
    """

    imported_torch_loss_func = v.ut.import_from_string(
        package="torch", module="nn", function=loss_function
    )()

    return imported_torch_loss_func
