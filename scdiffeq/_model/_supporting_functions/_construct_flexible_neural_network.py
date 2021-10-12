# _compose_neural_network.py
__module_name__ = "_compose_neural_network.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# package imports #
# --------------- #
import torch
import torch.nn as nn
import vintools as v
from collections import OrderedDict


def _enumerate_activation_functions(return_funcs=True, silent=True):

    """
    Enumerate over available activation functions within the torch.nn module.
    Convenience function.

    Parameters:
    -----------
    return_funcs
        default: False
        type: bool

    silent
        default: True
        type: bool

    Returns:
    --------
    returned_activation_functions
        list of available activation functions.
        Only returned if `return_funcs`==True.

    Notes:
    ------
    (1) Prints output of available activation functions.
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


def _check_pytorch_activation_function(activation_function):

    """
    Asserts that the passed activation function is valid within torch.nn library.
    Convenience function.

    Parameters:
    -----------
    activation_function

    Returns:
    --------
    None, potential assertion

    Notes:
    ------
    """

    assert activation_function.__class__.__name__ in _enumerate_activation_functions(
        return_funcs=True, silent=True
    ), _enumerate_activation_functions(return_funcs=False, silent=False)


def _construct_hidden_layers(
    neural_net, activation_function, hidden_layers, hidden_nodes
):

    neural_net[
        "{}_input".format(str(activation_function).strip("()"))
    ] = activation_function
    for n in range(0, hidden_layers):
        neural_net["hidden_layer_{}".format(n)] = nn.Linear(hidden_nodes, hidden_nodes)
        if n != (hidden_layers - 1):
            neural_net[
                "{}_{}".format(str(activation_function).strip("()"), n)
            ] = activation_function
        else:
            neural_net[
                "{}_output".format(str(activation_function).strip("()"))
            ] = activation_function

    return neural_net


def _construct_flexible_neural_network(
    in_dim=2, out_dim=2, layers=3, nodes=5, activation_function=nn.Tanh(),
):

    """
    Create a neural network given in/out dimensions and hidden unit dimenions.

    Parameters:
    -----------
    in_dim
        default: 2

    out_dim
        default: 2

    layers
        Number of fully connected torch.nn.Linear() hidden layers.
        default: 2

    nodes
        Number of nodes within a fully connected torch.nn.Linear() hidden layer.
        default: 5

    activation_function
        default: nn.Tanh()
        type: torch.nn.modules.activation.<func>

    Returns:
    --------
    nn.Sequential(neural_network)

    Notes:
    ------
    (1) Additional flexibility may be possible.
    """

    _check_pytorch_activation_function(activation_function)
    neural_net = OrderedDict()

    neural_net["input_layer"] = torch.nn.Linear(in_dim, nodes)
    neural_net = _construct_hidden_layers(neural_net, nn.Tanh(), layers, nodes)
    neural_net["output_layer"] = torch.nn.Linear(nodes, out_dim)

    return nn.Sequential(neural_net)
