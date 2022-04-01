
import licorice
import matplotlib.pyplot as plt
import numpy as np
import os
import vinplots


def _return_method_names(accuracy_df):
    
    idx_mask = accuracy_df.index.str.endswith("_predictions").fillna(False)
    existing_methods = accuracy_df.index[idx_mask].tolist()
    
    methods = []
    
    for i in existing_methods:
        methods.append(i.split("_")[0])
        
    methods = ["Smoothed\n'Ground Truth'"] + methods
    methods = methods + accuracy_df.index[4:].tolist()
    
    return methods


def _mkplot(
    nplots=3,
    ncols=3,
    width_ratios=[1, 0.2, 1],
    wspace=0.2,
    figsize=False,
    figsize_height=False,
    figsize_width=0.5,
):

    fig = vinplots.Plot()
    fig.construct(
        nplots=nplots,
        ncols=ncols,
        width_ratios=width_ratios,
        wspace=wspace,
        figsize=figsize,
        figsize_height=figsize_height,
        figsize_width=figsize_width,
    )
    fig.modify_spines(
        ax="all",
        spines_to_delete=["top", "right"],
        spines_to_move=["left"],
        spines_positioning_amount=10,
    )
    fig.AxesDict[0][1].remove()
    ax_r, ax_auroc = fig.AxesDict[0][0], fig.AxesDict[0][2]

    return fig, [ax_r, ax_auroc]

def _get_ymin(accuracy_df):
    
    """"""
    
    _df = accuracy_df.copy().dropna()
    min_auroc = _df[['auroc', 'auroc_masked']].values.min()
    min_r = _df[['r', 'r_masked']].values.min()
    
    if min_r <= 0.1:
        ymin_r = round(min_r - 0.1, 2)
    else:
        ymin_r = 0
        
    if min_auroc <=0.1:
        ymin_auroc = round(min_r - 0.1, 2)
    else:
        ymin_auroc = 0
        
    return [ymin_r, ymin_auroc]


def _format_axes(accuracy_df, ax_r, ax_auroc, y_labels, approx_prescient):
    
    methods = _return_method_names(accuracy_df)
    
    ymin_vals = _get_ymin(accuracy_df)

    for n, ax in enumerate([ax_r, ax_auroc]):
        ax.set_ylim(ymin_vals[n], 1)
        ax.set_xlim(-0.5, len(accuracy_df))
        ax.set_ylabel(y_labels[n])
        ax.set_xticks(range(len(accuracy_df.index)))
        ax.set_xticklabels(methods, ha="right", rotation=45)
        ax.hlines(
            y=approx_prescient[n],
            xmin=-0.5,
            xmax=len(accuracy_df) + 0.5,
            color="darkred",
            ls="--",
            zorder=5,
        )
        highlight_bar_height = 1 + abs(ymin_vals[n])
        ax.bar(x=0, bottom=ymin_vals[n], height=highlight_bar_height, zorder=0, color="dimgrey", alpha=0.1)
        ax.vlines(
            x=0.4, ymin=ymin_vals[n], ymax=1, color="dimgrey", ls="--", lw=1, alpha=0.8, zorder=1
        )

        
def _make_savename(save_prefix, savename):
    
    if not type(savename) == str:
        savename = ""
        
    if not (
        savename.endswith(".png")
        or savename.endswith(".svg")
        or savename.endswith(".pdf")
    ):
        savename = savename + ".svg"
    
    if savename == ".svg":
        return "".join([save_prefix, savename])
    else:
        return ".".join([save_prefix, savename])


def _save_fig(savename, outpath, layers, nodes, N, seed):
    
    save_prefix = "335cell_test_set.performance.compared.{}layers.{}nodes.{}N.seed{}".format(layers,
                                                                                               nodes,
                                                                                               N,
                                                                                               seed)
    save_prefix = os.path.join(outpath, save_prefix)
    if savename:
        savename = _make_savename(save_prefix, savename)
        print("\n\n{}: {}\n".format(licorice.font_format("Saving evaluation plot to", ["BOLD"]), savename))
        plt.savefig(savename)

def _plot_model_accuracy(
    accuracy_df,
    N,
    nodes,
    layers,
    seed,
    outpath,
    markersize=140,
    approx_prescient=[0.43, 0.72],
    model_color="#0A9396",
    y_labels=["Pearson's Rho", "AUROC"],
    savename=False,
    width_ratios=[1, 0.2, 1],
    wspace=0.2,
    figsize_width=0.5,
    figsize_height=1,
    figsize=False,
):

    """"""

    _accuracy_df = accuracy_df.copy()

    colors = ["#3A3B3C", "black", "black", "black"] + np.full(
        len(_accuracy_df.index[4:]), model_color
    ).tolist()

    fig, [ax_r, ax_auroc] = _mkplot(
        nplots=3,
        ncols=3,
        width_ratios=width_ratios,
        wspace=wspace,
        figsize=figsize,
        figsize_height=figsize_height,
        figsize_width=figsize_width,
    )

    for n, model in enumerate(_accuracy_df.index):
        _df = _accuracy_df.loc[_accuracy_df.index == model]
        ax_r.scatter(
            n, _df["r_masked"], c="white", edgecolor=colors[n], ls="--", s=markersize
        )
        ax_r.scatter(n, _df["r"], c=colors[n], s=markersize)
        ax_auroc.scatter(
            n,
            _df["auroc_masked"],
            c="white",
            edgecolor=colors[n],
            ls="--",
            s=markersize,
            zorder=20,
        )
        ax_auroc.scatter(
            n,
            _df["auroc"],
            c=colors[n],
            s=markersize,
            zorder=20,
        )

    _format_axes(_accuracy_df, ax_r, ax_auroc, y_labels, approx_prescient)
    plt.suptitle("335-cell 'Early Neu/Mo' Test Set (N={})".format(N), y=1.04)
    _save_fig(savename, outpath, layers, nodes, N, seed)
    plt.show()