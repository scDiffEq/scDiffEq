import vintools as v

from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import Colorbar
import matplotlib.font_manager

font = {"size": 12}
matplotlib.rc(font)
matplotlib.rcParams["font.sans-serif"] = "Arial"
matplotlib.rcParams["font.family"] = "sans-serif"

import matplotlib.pyplot as plt

def _gridspec_colorbar(fig, mappable, gridspec, label, label_rotation=0, labelpad=25):

    cbax = fig.add_subplot(gridspec)
    cb = Colorbar(
        ax=cbax,
        mappable=mappable,
        ticklocation="right",
    )
    cb.outline.set_visible(False)
    cb.set_label(label, rotation=0, labelpad=labelpad)

def _KernelDensity_plot_presets(
    DensityDict,
    plot_title="Kernel Density",
    x_lab="$x$",
    y_lab="$y$",
    title_fontsize=16,
    label_fontsize=14,
    figsize=(6, 5),
    savefigname=False,
    figure_legend_loc=0,
    **kwargs
):

    x = DensityDict["x"]
    y = DensityDict["y"]
    x_mesh = DensityDict["x_mesh"]
    y_mesh = DensityDict["y_mesh"]
    density = DensityDict["density"]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(ncols=2, nrows=1, width_ratios=[1, 0.08], wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])

    cmesh = ax.pcolormesh(x_mesh, y_mesh, density)
    ax.scatter(x, y, s=2, facecolor="white", label="Predicted Cells")

    spines = v.pl.ax_spines(ax)
    spines.set_color("grey")
    spines.delete(select_spines=["top", "right"])
    spines.set_position(position_type="axes", amount=-0.05)

    v.pl.set_minimal_ticks(ax, x_mesh, y_mesh)
    v.pl.legend(ax, loc=figure_legend_loc)
    ax.set_title(plot_title, y=1.05, fontsize=title_fontsize)
    ax.set_xlabel("$x$", size=label_fontsize)
    ax.set_ylabel("$y$", size=label_fontsize)
    plt.grid(zorder=0, c="lightgrey", alpha=0.5)

    _gridspec_colorbar(fig, mappable=cmesh, gridspec=gs[0, 1], label="Density")

    if savefigname:
        out = plt.savefig(savefigname)