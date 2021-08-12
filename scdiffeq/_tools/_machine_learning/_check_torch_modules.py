
# package imports #
# --------------- #
import vintools as v
import torch.nn as nn

def _enumerate_torchfunc_modules(
    self, torch_functions, silent=False, return_funcs=False
):

    """
    Prints and/or returns available functions for a given torch module.

    Parameters:
    -----------
    torch_functions
        type: torch.nn.modules.<function_type>

    silent
        default: False
        type: bool

    return_funcs
        default: False
        type: bool

    Returns:
    --------
    returned_functions [ optional ]
        list of available functions for a given torch module.
    """

    returned_functions = []
    if not silent:
        print(
            v.ut.format_pystring("Available activation functions:\n", ["BOLD", "RED"])
        )

    for n, func in enumerate(torch_functions):
        if not func.startswith("__"):
            if func not in self.not_torch_funcs:
                returned_functions.append(func)
                if n % 3 == 0:
                    if not silent:
                        print("{:<35}".format(func), end="\n")
                else:
                    if not silent:
                        print("{:<35}".format(func), end="\t")
    if return_funcs:
        return returned_functions


class _check_torch_modules:

    """
    check_torch_modules.activation()
    check_torch_modules.loss()
    """

    def __init__(
        self,
    ):

        self.nn_activation_funcs = nn.modules.activation.__dir__()
        self.nn_loss_funcs = nn.modules.loss.__dir__()

        self.not_torch_funcs = [
            "Tuple",
            "torch",
            "Tensor",
            "warnings",
            "Optional",
            "__name__",
            "__doc__",
            "__package__",
            "__loader__",
            "__spec__",
            "__file__",
            "__cached__",
            "__builtins__",
            "warnings",
        ]

    def activation(self, silent=False, return_funcs=False):

        """
        Parameters:
        -----------
        self
        
        silent[ optional ]
            default: False
        
        return_funcs[ optional ]
            default: False
        
        Returns:
        --------
        [ optional ] activation_funcs
        """

        activation_funcs = _enumerate_torchfunc_modules(
            self,
            torch_functions=self.nn_activation_funcs,
            silent=silent,
            return_funcs=return_funcs,
        )
        if return_funcs:
            return activation_funcs

    def loss(self, silent=False, return_funcs=False):

        """
        Parameters:
        -----------
        self
        
        silent[ optional ]
            default: False
        
        return_funcs[ optional ]
            default: False
        
        Returns:
        --------
        [ optional ] loss_funcs
        """

        loss_funcs = _enumerate_torchfunc_modules(
            self,
            torch_functions=self.nn_loss_funcs,
            silent=silent,
            return_funcs=return_funcs,
        )

        if return_funcs:
            return loss_funcs