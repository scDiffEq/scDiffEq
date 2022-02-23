import vinplots
import numpy as np
import pandas as pd


def _get_dict_best_training_epochs(results_dict):

    """get an ensemble of the best 5 training epochs"""

    BestResults = {}

    for key in results_dict.keys():
        BestResults[key] = {}
        n_epochs = results_dict[key].shape[0]
        top_means = results_dict[key].sort_values("mean").head(5)["mean"]
        BestResults[key]["mean"] = top_means.mean()
        BestResults[key]["std"] = top_means.std()
        BestResults[key]["n_epochs"] = n_epochs

    return BestResults


def _ensemble_best_training_epochs(ax, results_dict, plot=True):

    """"""

    BestResults = _get_dict_best_training_epochs(results_dict)
    res_df = pd.DataFrame(BestResults).T.sort_values("mean", ascending=False)
    res_df = res_df.reset_index().rename({"index": "composition"}, axis=1)
    comp_df = res_df.composition.str.split("layers_", expand=True)[1].str.split(
        ".nodes_", expand=True
    )
    comp_df.columns = ["layers", "nodes"]
    BestResults_df = pd.concat([res_df, comp_df], axis=1)

    if plot:
        _plot_barchart_best_results_ensembled(ax, BestResults_df)

    return BestResults_df


def _plot_barchart_best_results_ensembled(ax, BestResults_df):

    """"""
    x_pos = np.arange(len(BestResults_df))
    ax.bar(
        x=x_pos,
        height=BestResults_df["mean"],
        width=0.8,
        color="grey",
        alpha=1,
        edgecolor="black",
    )

    ax.errorbar(
        x=x_pos,
        y=BestResults_df["mean"],
        yerr=BestResults_df["std"],
        fmt="None",
        capsize=10,
        c="black",
    )

    gc = ax.set_xticks(x_pos)
    gc = ax.set_xticklabels(
        BestResults_df["composition"], fontsize=12, ha="right", rotation=30
    )


class _BestTrainingEnsemble:
    def __init__(self):

        self.fig = vinplots.Plot()
        self.fig.construct(nplots=1, ncols=1, figsize=1.2)
        self.fig.modify_spines(ax="all", spines_to_delete=["top", "right"])
        self.ax = self.fig.AxesDict[0][0]

    def plot(self, results_dict, return_dict=False):

        self.df = _ensemble_best_training_epochs(self.ax, results_dict, plot=True)
        if return_dict:
            return self.df


def _ensemble_best_training_results(results_dict, return_dict):

    """"""

    ensemble = _BestTrainingEnsemble()
    ensemble.plot(results_dict)

    if return_dict:
        return ensemble.df