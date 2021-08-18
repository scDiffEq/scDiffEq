# package imports #
# --------------- #
import vintools as v


def _choose_optimizer(self, optimizer_choice, learning_rate):

    """
    optimizer_choice
    
    learning_rate
    
    """

    optimizer_func = v.ut.import_from_string("torch", "optim", optimizer_choice,)
    optimizer = optimizer_func(self.network.parameters(), lr=learning_rate)

    return optimizer
