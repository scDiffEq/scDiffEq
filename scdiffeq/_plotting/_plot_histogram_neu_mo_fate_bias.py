
import numpy as np
import vinplots

def _mkplot(figsize):

    fig = vinplots.Plot()
    fig.construct(nplots=1, ncols=1, figsize=figsize)
    fig.modify_spines(
        ax="all",
        spines_to_delete=["top", "right"],
        spines_to_move=["left"],
        spines_positioning_amount=10,
    )
    ax = fig.AxesDict[0][0]

    
    return fig, ax

def _format_plot_ax(ax,
                    title="Fate Bias Scores",
                    ylab="Cell count",
                    xtick_labels = ["0.00\nMonocyte", "0.25", "0.50\nBipotent", "0.75", "1.00\nNeutrophil"]
                   ):
    
    ax.set_ylabel(ylab)
    ax.set_xticks([0, 2.5, 5, 7.5, 10])
    ax.set_xticklabels(xtick_labels)
    ax.set_title(title)

def _plot_histogram_neu_mo_fate_bias(scores,
                                     n_bins=10,
                                     title="Fate Bias Scores",
                                     figsize=1.2):

    fig, ax = _mkplot(figsize)

    color_scheme = vinplots.colors.BlueOrange(n_bins + 1)
    bins = np.histogram(scores, bins=n_bins, range=(0, 1))
    ax.bar(np.arange(len(bins[0])) + 0.5, bins[0], width=1, color=color_scheme)
    _format_plot_ax(ax, title)