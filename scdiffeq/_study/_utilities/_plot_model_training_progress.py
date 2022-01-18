import os
import vinplots
import pandas as pd

from ._get_best_model_path import _get_best_model_path

def _build_model_training_plot():

    """"""

    fig = vinplots.Plot()
    fig.construct(
        nplots=2,
        ncols=2,
        width_ratios=[1, 0.05],
        figsize_height=1.8,
        figsize_width=1,
    )
    fig.modify_spines(
        ax="all",
        spines_to_delete=["top", "right"],
        spines_positioning_amount=15,
        spines_to_move=["bottom", "left"],
    )
    ax1, ax2 = fig.AxesDict[0][0], fig.AxesDict[0][1]
    ax2.remove()

    return fig, ax1

def _plot_loss_curve(ax, log_df, best_model):

    ax.fill_between(
        x=log_df["epoch"],
        y1=0,
        y2=log_df["training_loss_4"],
        color="darkorange",
        alpha=0.2,
        label="Day 4",
    )

    ax.fill_between(
        x=log_df["epoch"],
        y1=log_df["training_loss_4"],
        y2=log_df["total_loss"],
        color="darkorange",
        alpha=0.6,
        label="Day 6",
    )

    ax.plot(log_df["epoch"], log_df["total_loss"], c="navy", lw=3, label="Total Loss")
    ax.set_title("Best model: epoch {}".format(best_model.epoch))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Wasserstein Distance")

    ax.vlines(
        x=int(best_model.epoch),
        ymin=0,
        ymax=log_df["total_loss"].max(),
        color="black",
        linestyles="--",
        label="Best model",
    )
    ax.legend(edgecolor="white", loc=[0.92, 0.7])
    
def _plot_model_training_progress(results_path, return_best_model=False):

    """"""

    best_model = _get_best_model_path(results_path)

    log_path = os.path.join(results_path, "status.log")
    log_df = pd.read_csv(log_path, sep="\t")

    fig, ax = _build_model_training_plot()

    _plot_loss_curve(ax, log_df, best_model)
    
    if return_best_model:
        return best_model