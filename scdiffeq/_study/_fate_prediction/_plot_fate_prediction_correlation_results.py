
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import vinplots

def _return_neu_vs_mo(adata, heldout=None, early=None):

    """
    By default, returns 1429 cells with annotated neu vs mo fates. These are cells that
    have any neutrophils or monocytes in their final fates. Thus, this subset
    is limited inherently to cells with a clonal barcode.

    heldout: [0, 1]

    early: [0, 1]
    """

    neu_vs_mo_df = adata.obs.loc[adata.obs["neu_vs_mo_percent"] >= 0]

    if not heldout == None:
        neu_vs_mo_df = neu_vs_mo_df.loc[neu_vs_mo_df["heldout_mask"] == heldout]
    if not early == None:
        neu_vs_mo_df = neu_vs_mo_df.loc[neu_vs_mo_df["early_neu_mo"] == early]

    neu_vs_mo = neu_vs_mo_df["neu_vs_mo_percent"]

    idx = neu_vs_mo.index
    val = neu_vs_mo.values

    return val, idx

def _calculate_correlation_with_observed_neu_mo(
    adata, method_prediction_key, heldout=None, early=None
):

    """"""

    observed, idx = _return_neu_vs_mo(adata, heldout=heldout, early=early)
    pred = adata[idx].obs[method_prediction_key]

    rho, statistic = pearsonr(pred, observed)
#     print(method_prediction_key, rho, statistic)

    return [rho, statistic]

def _build_correlation_results_plot(nplots, ncols, hspace, wspace, figsize_width):

    """"""

    fig = vinplots.Plot()
    fig.construct(
        nplots=nplots,
        ncols=ncols,
        hspace=hspace,
        wspace=wspace,
        figsize_width=figsize_width,
    )
    fig.modify_spines(ax="all", spines_to_delete=["top", "right"])

    return fig


def _plot_histogram_ground_truth_neu_mo(
    adata, ax, early_mask, heldout_mask, color_scheme
):

    """"""

    observed, idx = _return_neu_vs_mo(adata, early=early_mask, heldout=heldout_mask)

    bins = np.histogram(observed)
    ax.bar(range(len(bins[0])), bins[0], width=1, color=color_scheme)
    title_basis = (
        "Ground Truth Neu/Mo fate biases\nHeldout: {} | Early: {} | n_cells: {}"
    )
    ax.set_title(title_basis.format(heldout_mask, early_mask, len(idx)))
    ax.set_xticks([0, 9])
    ax.set_xticklabels(["Monocyte", "Neutrophil"])
    ax.set_xlabel("Fate Bias")
    ax.set_ylabel("Cell count")

    return [observed, idx]


def _get_prediction_columns(
    adata, ground_truth_key="smoothed_groundtruth_from_heldout", regex="predictions"
):

    """"""

    return [ground_truth_key] + adata.obs.filter(regex=regex).columns.tolist()


def _get_named_correlation_values(
    adata,
    heldout_mask,
    early_mask,
    ground_truth_key="smoothed_groundtruth_from_heldout",
    regex="predictions",
    add_in=False,
):

    pred_cols = _get_prediction_columns(adata, ground_truth_key, regex)

    corr_vals = []
    names = ["Smoothed\nGround Truth"]
    for n, col in enumerate(pred_cols):
        if n != 0:
            names.append(col.split("_")[0])
        corr, stat = _calculate_correlation_with_observed_neu_mo(
            adata, col, heldout_mask, early_mask
        )
        corr_vals.append(corr)
    if add_in:
        names.append(add_in[0])
        corr_vals.append(add_in[1])            

    return corr_vals, names


def _plot_correlation_results_scatter(ax, correlation_values, method_names):

    """Assumes the first value is the ground truth"""

    ax.scatter(
        0, correlation_values[0], c="dimgrey", s=75, edgecolor="k", linewidth=0.5
    )
    ax.scatter(
        range(1, 1 + len(correlation_values[1:])),
        correlation_values[1:],
        c="red",
        s=75,
        edgecolor="k",
        linewidth=0.5,
    )
    ax.set_xlim(-0.5, len(correlation_values))
    ax.set_ylim(0, 1)
    ax.vlines(x=0.5, ymin=0, ymax=1, color="black", ls="--")
    ax.set_xticks(range(len(correlation_values)))
    ax.set_xticklabels(method_names)
    ax.set_title("Correlation with Ground\nTruth N/M fate bias")
    ax.set_ylabel("Pearson's \u03C1")


def _plot_fate_prediction_correlation_results(
    adata,
    ncols=2,
    hspace=0.6,
    wspace=0.6,
    figsize_width=0.7,
    figsize_height=1.1,
    ground_truth_key="smoothed_groundtruth_from_heldout",
    pred_regex="predictions",
    save=True,
    savename="groundtruth.neu_vs_mo_percent.divisions.svg",
    add_in=False
):

    """"""

    color_scheme = vinplots.palettes.BlueOrange(10)
    early_mask_toggles = [None, None, None, 0, 1, 1, 0, 0, 1]
    heldout_mask_toggles = [None, 0, 1, None, None, 0, 1, 0, 1]
    nplots = len(early_mask_toggles) + len(heldout_mask_toggles)

    fig = _build_correlation_results_plot(nplots, ncols, hspace, wspace, figsize_width)

    for n, [e, h] in enumerate(zip(early_mask_toggles, heldout_mask_toggles)):
        axes = fig.AxesDict[n]
        _plot_histogram_ground_truth_neu_mo(adata, axes[0], e, h, color_scheme)
        correlation_values, method_names = _get_named_correlation_values(
            adata,
            heldout_mask=h,
            early_mask=e,
            ground_truth_key=ground_truth_key,
            regex=pred_regex,
            add_in=add_in,
        )
        _plot_correlation_results_scatter(axes[1], correlation_values, method_names)
    if save:
        plt.savefig(savename)