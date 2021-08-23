import numpy as np
import matplotlib.pyplot as plt

# from IPython import display
import vintools as v


def _get_x_axis_plot_train_validation(adata, groupsize=1):

    """ """
    len_train_loss = len(adata.uns["loss"]["train_loss"])
    len_valid_loss = len(adata.uns["loss"]["valid_loss"])

    validation_frequency = adata.uns["validation_frequency"]

    x_range_train = np.arange(groupsize, (len_train_loss + groupsize), groupsize)

    x_range_valid = (
        np.arange(groupsize, (len_valid_loss + groupsize), groupsize)
        * validation_frequency
    )

    x_range_train_adjusted = np.linspace(1, x_range_train.max(), len(x_range_train))
    x_range_valid_adjusted = np.linspace(1, x_range_valid.max(), len(x_range_valid))

    return x_range_train_adjusted, x_range_valid_adjusted


def _plot_smoothed_training(
    adata, groupsize=5, silence_stdev=False, grid=True, save_path=False,
):

    """
    Plot training.

    Parameters:
    -----------
    adata
        AnnData

    groupsize
        default: 5
        type: int

    silence_stdev
        default: False

    grid
        default: True

    Returns:
    --------
    None
        plots inputs

    Notes:
    ------
    if the plot is being updated live, the final point will float as it is the mean of a bucket of points,
    until that bucket is completed.
    """

    train_loss = adata.uns["loss"]["train_loss"]
    valid_loss = adata.uns["loss"]["valid_loss"]

    x_range_train, x_range_valid = _get_x_axis_plot_train_validation(
        adata, groupsize=groupsize
    )

    fig, ax = plt.subplots()

    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSELoss")
    ax.set_title("Training progress: {} epochs".format(str(len(train_loss))), y=1.05)
    
#     detached_train_loss = []
#     detached_valid_loss = []
#     for ti in train_loss:
#         detached_train_loss.append(ti.cpu())
#     for vl in valid_loss:
#         detached_valid_loss.append(vl.cpu())
        
    
    smoothed_mean_train, smoothed_stdev_train = v.ut.smooth(
        unpartitioned_items=train_loss, groupsize=groupsize
    )
    smoothed_mean_valid, smoothed_stdev_valid = v.ut.smooth(
        unpartitioned_items=valid_loss, groupsize=groupsize
    )

    # plot training loss
    plt.plot(x_range_train, smoothed_mean_train, c="navy", zorder=2, label="training")
    if not silence_stdev:
        hi, low = (
            smoothed_mean_train + smoothed_stdev_train,
            smoothed_mean_train - smoothed_stdev_train,
        )
        plt.fill_between(x_range_train, hi, low, color="navy", alpha=0.1, zorder=1)

    # plot validation loss
    plt.plot(
        x_range_valid, smoothed_mean_valid, c="darkorange", zorder=2, label="validation"
    )
    if not silence_stdev:
        hi, low = (
            smoothed_mean_valid + smoothed_stdev_valid,
            smoothed_mean_valid - smoothed_stdev_valid,
        )
        plt.fill_between(
            x_range_valid, hi, low, color="darkorange", alpha=0.25, zorder=1
        )

    if grid:
        plt.grid(zorder=0)

    v.pl.legend(ax, loc=1)
    # save and display plot
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def _plot_loss(adata, groupsize, save_path):

    """"""

    _plot_smoothed_training(
        adata, groupsize=groupsize, save_path=save_path,
    )


#     display.clear_output(wait=True)
#     display.display(plt.gcf())
