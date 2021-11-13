
import matplotlib.pyplot as plt
import numpy as np
import os

def _format_valid_predictions_for_plotting(evaluator, group):

    x, y = evaluator.y[group].detach().numpy()[:, 0].T, evaluator.y[group].detach().numpy()[:, 1].T
    x_, y_ = (
        evaluator.pred_y[group].detach().numpy()[:, :, 0],
        evaluator.pred_y[group].detach().numpy()[:, :, 1],
    )

    return [x, y, x_, y_]

def _tile_time(time, pred):
    return np.tile(time, pred.shape[0])

def _size_adjust_loss_values(loss_vector, GroupSizesDict, group):
    return np.array(loss_vector)*100 / GroupSizesDict[group]

def _scale_loss_values_for_ploting(TrainingMonitor, validator):
    
    size_adjusted_train_loss = _size_adjust_loss_values(
        TrainingMonitor.train_loss, validator.GroupSizesDict, group="train"
    )
    size_adjusted_valid_loss = _size_adjust_loss_values(
        TrainingMonitor.valid_loss, validator.GroupSizesDict, group="valid"
    )
        
    scaled_train_loss = size_adjusted_train_loss / size_adjusted_train_loss.max()
    scaled_valid_loss = size_adjusted_valid_loss / size_adjusted_valid_loss.max()
    
    return scaled_train_loss, scaled_valid_loss

def _save_training_validation_update_plot(
    results_dir,
    epoch,
    fig_savename="training_validation_update",
    filetype_extension=".pdf",
    return_path=False,
):

    """ """

    training_validation_update_path = (
        os.path.join(
            results_dir,
            "scdiffeq/imgs/training",
            "{}.epoch_{}".format(fig_savename, epoch),
        )
        + filetype_extension
    )

    v.ut.mkdir_flex(os.path.dirname(training_validation_update_path))
    plt.savefig(training_validation_update_path, bbox_inches="tight")

    if return_path:
        return training_validation_update_path
    
def _set_ax_labels_title(ax, title, title_fontsize, x_label, y_label):

    """"""

    ax.set_title(title, fontsize=title_fontsize, y=1.06)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def _plot_ax(
    ax, x, y, colorby, scatter_size, title, title_fontsize, x_label, y_label, zorder
):

    _set_ax_labels_title(ax, title, title_fontsize, x_label, y_label)
    ax.scatter(x, y, s=scatter_size, c=colorby, zorder=zorder)
    ax.scatter(x, y, s=scatter_size, c="lightgrey", alpha=0.75, zorder=zorder - 1)


def _prepare_plot_coordinates(TrainingMonitor, validator, scaled_train_loss, binsize):

    BinnedMeanStd, max_bin = _prepare_binned_loss(scaled_train_loss, binsize)
    x_range = np.arange(len(BinnedMeanStd["mean"])) * binsize

    y_mean = np.array(BinnedMeanStd["mean"])
    y1 = y_mean + np.array(BinnedMeanStd["std"])
    y2 = y_mean - np.array(BinnedMeanStd["std"])

    return x_range, y_range, y_mean, y1, y2


def _plot_loss_curve(ax, HyperParameters, TrainingMonitor, validator, binsize):

    """"""

    scaled_train_loss, scaled_valid_loss = _scale_loss_values_for_ploting(
        TrainingMonitor, validator
    )
    x_range, y_range, y_mean, y1, y2 = _prepare_plot_coordinates(
        TrainingMonitor, validator, scaled_train_loss, binsize
    )

    ax.fill_between(x_range, y1, y2, color="navy", alpha=0.2, zorder=1)
    ax.plot(x_range, y_mean, c="navy", zorder=2, label="Training")

    x_range_valid = (
        np.arange(len(scaled_valid_loss)) * HyperParameters.validation_frequency
    )
    ax.plot(x_range_valid, scaled_valid_loss, c="orange", zorder=2, label="Validation")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("{}".format(HyperParameters.loss_function))

    ax.legend(
        loc=1,
        edgecolor="w",
    )


def _plot_validation_training_update(
    adata,
    validator,
    TrainingMonitor,
    HyperParameters,
    binsize,
    title_fontsize=12,
    zorder=1,
    scatter_size=10,
    y=1.06,
    x_label="$x$",
    y_label="$y$",
):

    """

    Notes:
    ------
    (1) The `plot.construct_layout` step is already relatively size-optimized for a jupyter
        notebook but may be case-restricted in practice.

    (2)
    """

    plot_titles = [
        "Training Data",
        "Training Predictions",
        "Training Predictions Overlaid",
        "Training and Validation Loss",
        "Validation Data",
        "Validation Predictions",
        "Validation Predictions Overlaid",
    ]

    colorby = np.sort(adata.obs.time.unique())

    plot = v.pl.ScatterPlot()
    plot.construct_layout(
        nplots=8,
        ncols=4,
        grid_hspace=0.4,
        width_ratios=np.ones(4),
        figsize_height=1.4,
        figsize_width=1,
    )
    plot.style()

    # rename AxesDict.keys() for accounting / sensible subsetting
    plot.AxesDict["train"] = plot.AxesDict.pop(0)
    plot.AxesDict["valid"] = plot.AxesDict.pop(1)

    plot_n = 0
    for group in plot.AxesDict.keys():
        [x, y, x_, y_] = _format_valid_predictions_for_plotting(validator, group=group)
        t = _tile_time(validator.t, validator.pred_y[group])
        title = plot_titles[plot_n]
        for ax in plot.AxesDict[group].values()[:2]:
            if group == "train" and ax == 3:
                _plot_loss_curve(
                    ax, HyperParameters, TrainingMonitor, validator, binsize
                )
            else:
                _plot_ax(
                    ax,
                    x,
                    y,
                    t,
                    scatter_size,
                    title,
                    title_fontsize,
                    x_label,
                    y_label,
                    zorder,
                )
                plot_n += 1
    plot.AxesDict["valid"][3].remove()
    _save_training_validation_update_plot(results_dir, 
                                          epoch, 
                                          fig_savename="training_validation_update", 
                                          filetype_extension=".pdf", 
                                          return_path=False,)

    return plot