
# package imports #
# --------------- #
import torch.nn as nn
import torch.optim as optim

# local imports #
# ------------- #
from .._machine_learning._choose_optimizer import _choose_optimizer
from .._machine_learning._choose_loss_function import _choose_loss_function
from .._machine_learning._RunningAverageMeter import _RunningAverageMeter


def _preflight(
    self,
    adata,
    validation_frequency=20,
    visualization_frequency=20,
    loss_function="MSELoss",
    optimizer="RMSprop",
    learning_rate=1e-3,
):

    """
    Sets the following parameters required for learning a neural ODE.
    Defines the following:
        (1) loss function
        (2) optimizer
        (3) learning rate
        (4) validation frequency
        (5) visualization frequency

    Parameters:
    -----------
    self
    adata
    validation_frequnecy
    visualization_frequency
    loss_function
    optimizer
    learning_rate

    Returns:
    --------
    None
        all parameters and setup are modified in-place.

    Notes:
    ------
    This step must be run prior to learning an ODE using the neural ODE module.
    """

    # setup parameterizables
    self.learning_rate = learning_rate
    self.validation_frequency = adata.uns["validation_frequency"] = validation_frequency
    self.visualization_frequency = adata.uns[
        "visualization_frequency"
    ] = visualization_frequency
    self.loss_function = adata.uns["loss_func"] = _choose_loss_function(loss_function)
    self.optimizer = adata.uns["optimizer"] = _choose_optimizer(
        self, optimizer, learning_rate
    )

    # setup loss tracking
    adata.uns["loss"] = {}
    adata.uns["loss"]["train_loss"] = []
    adata.uns["loss"]["valid_loss"] = []
    self.loss = adata.uns["loss"]
    
    # setup RunningAverageMeter
    adata.uns["RunningAverageMeter"] = _RunningAverageMeter(0.97)
    
    # sync adata with class object
    self.adata = adata