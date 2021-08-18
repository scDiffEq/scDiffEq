# package imports #
# --------------- #
import numpy as np
import vintools as v
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager

font = {"size": 12}
matplotlib.rc(font)
matplotlib.rcParams["font.sans-serif"] = "Arial"
matplotlib.rcParams["font.family"] = "sans-serif"


def _simulation_plot_presets(
    x,
    y,
    plot_title="Simulation",
    x_lab="$x$",
    y_lab="$y$",
    title_fontsize=16,
    label_fontsize=14,
    figsize=(6, 5),
    **kwargs
):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    spines = v.pl.ax_spines(ax)
    spines.set_color("grey")
    spines.delete(select_spines=["top", "right"])
    spines.set_position(position_type="axes", amount=-0.05)

    v.pl.set_minimal_ticks(ax, x, y)

    ax.set_title(plot_title, y=1.05, fontsize=title_fontsize)
    ax.set_xlabel("$x$", size=label_fontsize)
    ax.set_ylabel("$y$", size=label_fontsize)
    ax.scatter(x, y, **kwargs)
    plt.grid(zorder=0, c="lightgrey", alpha=0.5)


def _plot(self, c="time", savefigname=None, **kwargs):

    X = self.adata.X
    if c == "time":
        c = self.adata.obs.time

    _simulation_plot_presets(X[:, 0], X[:, 1], c=c, **kwargs)
    if savefigname:
        out = plt.savefig(savefigname)
