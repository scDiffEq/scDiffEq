
import licorice
import numpy as np
import scdiffeq as sdq
import torch


def _load_trained_model(model_path, in_dim, out_dim, layers, nodes, device):

    model = sdq.scDiffEq(
        in_dim=in_dim, out_dim=out_dim, layers=layers, nodes=nodes, device=device
    )
    model.network_model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def _get_best_model_one_seed(
    BestModelsDict, name, in_dim, out_dim, subset_df, seed, device
):

    """"""

    best_model_path = subset_df["best_model_path"].iloc[seed]
    best_model_epoch = subset_df["best_model_epoch"].iloc[seed]

    BestModelsDict[name][
        "seed_{}.epoch_{}".format(seed, int(best_model_epoch))
    ] = _load_trained_model(
        best_model_path,
        in_dim,
        out_dim,
        layers=subset_df["layers"].iloc[seed],
        nodes=subset_df["nodes"].iloc[seed],
        device=device,
    )

    return BestModelsDict


def _get_best_model_each_seed(
    BestModelsDict, subset_df, group, in_dim, out_dim, device
):

    """"""

    if np.all(subset_df["best_model_path"].values != None):

        name = "layers{}.nodes{}".format(group[0], group[1])
        BestModelsDict[name] = {}

        for seed in range(len(subset_df)):
            BestModelsDict = _get_best_model_one_seed(
                BestModelsDict, name, in_dim, out_dim, subset_df, seed, device
            )

    return BestModelsDict


def _load_best_models(hp_table, in_dim=50, out_dim=50, device ="cpu"):

    BestModelsDict = {}
    for group, subset_df in hp_table.groupby(["layers", "nodes"]):
        for seed in range(len(subset_df)):
            BestModelsDict = _get_best_model_each_seed(
                BestModelsDict, subset_df, group, in_dim, out_dim, device
            )

    return BestModelsDict


def _make_model_dict(best_models):

    ModelDict = {}
    for model_setup in best_models.keys():
        for version in best_models[model_setup].keys():
            layers, nodes = model_setup.split("layers")[-1].split(".nodes")
            layers, nodes = int(layers), int(nodes)
            ModelDict["{}.{}".format(layers, nodes)] = best_models[model_setup][version]

    return ModelDict


def _get_available_loaded_best_models(best_models):

    ModelDict = _make_model_dict(best_models)
    print(licorice.font_format("\nAvailable Models:", ["BOLD"]), end=" ")
    for key in ModelDict.keys():
        if key != list(ModelDict.keys())[-1]:
            print("{}".format(key), end=", ")
        else:
            print("{}".format(key))

    return ModelDict