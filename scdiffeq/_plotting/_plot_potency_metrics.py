import os
import vintools as v
from ._saveplot import _saveplot
import matplotlib.pyplot as plt


def _subplot_fig_ax_presets(ax, title, x_lab, y_lab):

    ax.set_title(title, fontsize=15)
    ax.set_xlabel(x_lab, fontsize=14)
    ax.set_ylabel(y_lab, fontsize=14)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


def _plot_potency_metrics(adata, save=False, plot_savename=None, size=(12, 5)):

    fig = plt.figure(figsize=size)

    ax1 = fig.add_subplot(1, 2, 1)
    _subplot_fig_ax_presets(
        ax1, title="Potency Metrics", x_lab="Normalized Value", y_lab="Count"
    )
    plt.subplot(1, 2, 1)
    a = plt.hist(
        adata.obs.gene_count / adata.obs.gene_count.max(),
        color=v.pl.vin_colors()[9],
        alpha=0.75,
        label="gene count",
        bins=50,
    )

    b = plt.hist(
        adata.obs.potency,
        bins=50,
        color=v.pl.vin_colors()[2],
        alpha=0.75,
        label="potency",
    )
    plt.legend(
        markerscale=3,
        edgecolor="w",
        fontsize=14,
        handletextpad=None,
        bbox_to_anchor=(0.5, 0.0, 0.80, 1),
    )

    ax2 = fig.add_subplot(1, 2, 2)
    _subplot_fig_ax_presets(
        ax2, title="Cell Potency", x_lab="SPRING-x", y_lab="SPRING-y"
    )
    plt.scatter(
        adata.obsm["X_spring"][:, 0],
        adata.obsm["X_spring"][:, 1],
        c=adata.obs.potency,
        s=1,
        cmap="viridis",
    )
    plt.tight_layout()
    plt.colorbar()

    if save == True:
        if plot_savename == None:
            plot_savename = "gene_count_potency_histogram_dimensional_reduction.png"
        _saveplot(save_dir=os.getcwd(), save_name=plot_savename)

    plt.show()
