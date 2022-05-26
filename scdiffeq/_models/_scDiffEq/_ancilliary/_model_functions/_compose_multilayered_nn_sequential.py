
__module_name__ = "_compose_multilayered_nn_sequential.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import torch
from collections import OrderedDict


def _parse_activation_function_dict(activation_function_dict):

    items = list(activation_function_dict.items())[0]
    name = items[0]
    func = items[1]

    return name, func


def _compose_multilayered_nn_sequential(
    nodes_by_layer, activation_function_dict, dropout
):

    neural_net = OrderedDict()
    activation_func_name, activation_func = _parse_activation_function_dict(
        activation_function_dict
    )

    for i in range(len(nodes_by_layer) - 1):
        neural_net["{}_layer".format(i)] = torch.nn.Linear(
            nodes_by_layer[i], nodes_by_layer[i + 1]
        )
        if i != len(nodes_by_layer) - 2:
            if dropout:
                neural_net["{}_dropout".format(i)] = torch.nn.Dropout(dropout)
            neural_net["{}_{}".format(i, activation_func_name)] = activation_func

    return torch.nn.Sequential(neural_net)