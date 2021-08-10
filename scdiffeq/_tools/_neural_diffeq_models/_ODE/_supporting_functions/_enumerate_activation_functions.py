
# package imports #
# --------------- #
import vintools as v
import torch.nn as nn

def _enumerate_activation_functions(return_funcs=True, silent=True):

    """
    Parameters:
    -----------
    return_funcs
        default: False

    Returns:
    --------
    activ_funcs
        list of available activation functions.

    Printed output of available activation functions.
    """

    not_activation_funcs = ["Tuple", "torch", "Tensor", "warnings", "Optional"]

    activ_funcs = nn.modules.activation.__dir__()
    if not silent:
        print(
            v.ut.format_pystring("Available activation functions:\n", ["BOLD", "RED"])
        )

    returned_activation_functions = []

    for n, func in enumerate(activ_funcs):
        if not func.startswith("__"):
            if func not in not_activation_funcs:
                returned_activation_functions.append(func)
                if n % 3 == 0:
                    if not silent:
                        print("{:<35}".format(func), end="\n")
                else:
                    if not silent:
                        print("{:<35}".format(func), end="\t")
                        
    if return_funcs:
        return returned_activation_functions