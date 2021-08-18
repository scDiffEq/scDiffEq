import matplotlib.pyplot as plt


def presets_for_plotting_multiple_trajectories(ax, title, x, y, xlab, ylab):

    ax.set_xlabel(xlab, fontsize=15)
    ax.set_ylabel(ylab, fontsize=15)
    ax.set_title(title, fontsize=20)
    ax.plot(x, y, marker="o", linewidth=2, markersize=4)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    return ax


def plot_simulated_trajectories(adata, title="Simulated Data", figsize=(10, 8)):

    """"""

    figure = plt.figure(figsize=figsize)
    ax = figure.add_subplot(1, 1, 1)

    data_by_trajectory = adata.X.reshape(
        adata.uns["number_of_trajectories"],
        adata.uns["number_of_timepoints"],
        adata.X.shape[1],
    )

    for i in range(adata.uns["number_of_trajectories"]):
        presets_for_plotting_multiple_trajectories(
            ax,
            title,
            data_by_trajectory[i, :, 0],
            data_by_trajectory[i, :, 1],
            "$x$",
            "$y$",
        )
