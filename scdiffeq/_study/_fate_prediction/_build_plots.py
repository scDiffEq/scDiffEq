
import vinplots

def _build_plots(n_preds):

    fig = vinplots.Plot()

    if n_preds > 3:
        nplots = 12
        height_ratios = [1, 1, 0.15]
        bottom_row = 2
    else:
        nplots = 5
        height_ratios = [1, 0.15]
        bottom_row = 1
    fig.construct(
        nplots=nplots,
        ncols=4,
        width_ratios=[1, 1, 1, 1],
        height_ratios=height_ratios,
        hspace=0.2,
        figsize_width=1.2,
    )  # limited but fine for now
    fig.modify_spines(ax="all", spines_to_delete=["top", "bottom", "right", "left"])
    fig.AxesDict[bottom_row][0].set_xticks([])
    fig.AxesDict[bottom_row][0].set_yticks([])
    for i in fig.AxesDict[bottom_row]:
        if i > 0:
            fig.AxesDict[bottom_row][i].remove()

    axes = []
    for i in fig.AxesDict:
        for j in range(len(fig.AxesDict[i])):
            axes.append(fig.AxesDict[i][j])

    for n, ax in enumerate(axes):
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, axes, bottom_row
