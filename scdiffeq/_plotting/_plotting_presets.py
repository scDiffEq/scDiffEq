import matplotlib.pyplot as plt


def _presets_for_plotting_multiple_trajectories(ax, title, x, y, xlab, ylab):

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


def _single_fig_presets(
    title,
    x_lab,
    y_lab,
    size=(10, 8),
    title_fontsize=20,
    title_adjustment_factor=1.1,
    axis_label_fontsize=15,
):

    """
    presets for one single figure to look nice
    """

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=title_fontsize, y=title_adjustment_factor)
    ax.set_xlabel(x_lab, fontsize=axis_label_fontsize)
    ax.set_ylabel(y_lab, fontsize=axis_label_fontsize)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    return fig, ax


def _subplot_presets(ax, x, y, xlab, ylab, size, color, alpha):

    """"""

    ax.set_xlabel(xlab, fontsize=15)
    ax.set_ylabel(ylab, fontsize=15)
    ax.scatter(x, y, c=color, cmap="viridis", s=size, alpha=alpha)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    return ax


def _dual_plot_presets(
    title,
    left_x,
    left_y,
    right_x,
    right_y,
    x_lab_left,
    x_lab_right,
    y_lab_left,
    y_lab_right,
    size_ratio=1,
):

    fig, axes = plt.subplots(
        1, 2, figsize=(12 * size_ratio, 6 * size_ratio), facecolor="white"
    )

    fig.suptitle(title, y=1.02, fontsize=20, fontweight="semibold")

    ax1, ax2 = axes[0], axes[1]

    subplot_presets(ax1, left_x, left_y, x_lab_left, y_lab_left, size)
    subplot_presets(ax2, right_x, right_y, x_lab_right, y_lab_right, size)


def _annotate_scatterplot(
    points_to_be_annotated,
    labels,
    magnitude=[1, 1],
    direction=[1, 1],
    point_offset=None,
):

    """
    points_to_be_annotated
        list of coordinate points to be annotated.
    labels
        list of str labels ordered to match the points to be annotated.
        
    magnitude (optional)
    
    direction (optional)
        x,y tuple
    """

    for i, point in enumerate(points_to_be_annotated):

        x, y = point[0] + point_offset, point[1] + point_offset

        offset_x, offset_y = (
            x + magnitude[i][0] * direction[i][0],
            y + magnitude[i][1] * direction[i][1],
        )

        plt.annotate(
            labels[i],
            xy=(x, y),
            xytext=(offset_x, offset_y),
            arrowprops=dict(arrowstyle="-"),
        )
