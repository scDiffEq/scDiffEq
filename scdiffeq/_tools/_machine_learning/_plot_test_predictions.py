# package imports #
# --------------- #
import vintools as v
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# bbox_to_anchor=(0.5, 0.0, 0.70, 1),


def _annotate_legend(
    legend_ax, label_ax, markerscale=1, fontsize=12, loc=2, bbox=None, **kwargs
):

    """"""
    h, l = label_ax.get_legend_handles_labels()

    legend_ax.legend(
        h,
        l,
        markerscale=markerscale,
        edgecolor="w",
        fontsize=fontsize,
        handletextpad=None,
        loc=2,
    )

    legend_spines = v.pl.ax_spines(legend_ax)
    legend_spines.delete()

    legend_ax.set_xticks([])
    legend_ax.set_yticks([])


def _plot_predicted_test_data(
    adata, x_label="$x$", y_label="$y$", figsize=(6, 5.5), save_path=False,
):

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(ncols=2, nrows=1, width_ratios=[1, 0.08], wspace=0.1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title("Test data: predictions", y=1.05, fontsize=14)

    plt.grid(zorder=0, c="lightgrey", alpha=0.5)

    y_pred = adata.uns["test_y_predicted"]
    y = adata.uns["test_y"]

    test_trajs = adata.uns["test_trajectories"]

    for [key, value] in test_trajs.items():
        t_len = len(test_trajs.get(key).t)
        break

    time_vector = np.zeros([len(test_trajs), t_len])

    for n, key in enumerate(test_trajs.keys()):
        time_vector[n] = test_trajs[key].t

    for i in range(len(y)):
        ax.plot(y[i, :, 0], y[i, :, 1], c="lightgrey", zorder=1)
        ax.plot(y_pred[i, :, 0], y_pred[i, :, 1], c="black", zorder=2)

        ax.scatter(y[i, :, 0], y[i, :, 1], c="lightgrey", zorder=3, s=8, label="Data")
        ax.scatter(
            y_pred[i, :, 0],
            y_pred[i, :, 1],
            c=time_vector[i],
            zorder=4,
            s=8,
            label="Predicted",
        )

    spines = v.pl.ax_spines(ax)

    spines.set_color("grey")
    spines.delete(select_spines=["top", "right"])
    spines.set_position(position_type="axes", amount=-0.05)

    legend_ax = fig.add_subplot(gs[0, 1])
    legend_ax.scatter([], [], label="Data", c="lightgrey")
    legend_ax.scatter([], [], label="Predicted", c="black")
    _annotate_legend(legend_ax, legend_ax)
    all_x, all_y = (
        torch.vstack([y, y_pred])[:, :, 0],
        torch.vstack([y, y_pred])[:, :, 1],
    )

    # save and display plot
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
