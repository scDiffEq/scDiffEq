import pandas as pd


def _count_network_parameters(network):

    """Count the parameters in a neural network"""

    param_count = 0

    for parms in network.parameters():
        param_count += parms.numel()

    return param_count


def _count_parameters_all_models(best_models, silent=True):

    """For neural network models in a dict (key: model_name, value: model), return a DataFrame of nodes/layers/parameters."""

    ParameterCounts = {}
    ParameterCounts["layers"] = []
    ParameterCounts["nodes"] = []
    ParameterCounts["parameters"] = []

    for key, model in best_models.items():
        layers, nodes = (
            key.split(".")[0],
            key.split(".")[1],
        )
        params = _count_network_parameters(model.network_model)
        ParameterCounts["layers"].append(layers)
        ParameterCounts["nodes"].append(nodes)
        ParameterCounts["parameters"].append(params)

        if not silent:
            message = "Layers {}, Nodes: {}, Params: {}"
            print(message.format(layers, nodes, params))

    return pd.DataFrame(ParameterCounts).sort_values("parameters")