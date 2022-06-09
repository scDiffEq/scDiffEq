
__module_name__ = "_model.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import vinplots


def _build_plot(
    nplots,
    ncols=4,
    figsize=1.5,
    figsize_width=1,
    figsize_height=1,
    hspace=0.18,
    wspace=0,
    width_ratios=False,
    height_ratios=False,
    color=False,
    spines_to_color=False,
    spines_to_delete=["top", "right"],
    spines_to_move=False,
    spines_positioning="outward",
    spines_positioning_amount=0,
):

    if nplots < ncols:
        ncols = nplots

    fig = vinplots.Plot()

    fig.construct(
        nplots,
        ncols,
        figsize_width,
        figsize_height,
        figsize,
        hspace,
        wspace,
        width_ratios,
        height_ratios,
    )

    fig.modify_spines(
        "all",
        color,
        spines_to_color,
        spines_to_delete,
        spines_to_move,
        spines_positioning,
        spines_positioning_amount,
    )

    return fig, fig.AxesDict