
# _choose_optimizer.py
__module_name__ = "_choose_optimizer.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# package imports #
# --------------- #
import vintools as v


def _choose_optimizer(network_model, optimizer_choice, learning_rate):
    
    """
    Choose an optimizer from torch.optim (calling by string), given a learning_rate and network_model.
    
    Parameters:
    -----------
    network_model
        Neural network formulated differential equation model
        type: scdiffeq.Neural_Differential_Equation
    
    optimizer_choice
        Choice of torch optimizer
        type: str
    
    learning_rate
        type: float
    
    Returns:
    --------
    optimizer
    
    Notes:
    ------
    (1) Uses `v.ut.import_from_string("torch", "optim", optimizer_choice)` to 
        flexibly return an optimizer.
    """

    optimizer_func = v.ut.import_from_string("torch", "optim", optimizer_choice,)
    optimizer = optimizer_func(network_model.parameters(), lr=learning_rate)

    return optimizer
