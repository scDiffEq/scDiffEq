import vintools as v
import numpy as np

import matplotlib.font_manager

font = {"size": 12}
matplotlib.rc(font)
matplotlib.rcParams["font.sans-serif"] = "Arial"
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import Colorbar

from .._tools._machine_learning._get_2d_meshgrid_dydt import _get_2d_meshgrid_dydt

def _plot_meshgrid_vector_field(
    adata,
    ODE_key='ODE',
    figsize=(6, 5.5),
    title_y_adj=1.05,
    plot_title="Drift plot",
    bins=25,
    cmap="viridis",
    save_path=False,
    **stream_plot_kwargs
):

    """"""
    
    ODE = adata.uns[ODE_key]

    bounds = v.ut.get_data_bounds(adata.X)
    x_range = np.abs(np.subtract(bounds["x"]["min"], bounds["x"]["max"]))
    y_range = np.abs(np.subtract(bounds["y"]["min"], bounds["y"]["max"]))

    x, y, dydt, velo_mag = _get_2d_meshgrid_dydt(adata.X, ODE, bins)

    fig = plt.figure(figsize=figsize)
    gridspec = GridSpec(nrows=1, ncols=2, width_ratios=[1, 0.08], wspace=0.1)

    #### streamplot ####
    ax_stream = fig.add_subplot(gridspec[0, 0])
    stream = ax_stream.streamplot(
        y, x, dydt[:, :, 0], dydt[:, :, 1], color=velo_mag, **stream_plot_kwargs
    )
    stream_spines = v.pl.ax_spines(ax_stream)
    stream_spines.set_color("grey")
    stream_spines.delete(select_spines=["top", "right"])
    stream_spines.set_position(position_type="axes", amount=-0.05)
    v.pl.set_minimal_ticks(ax_stream, y, x, round_decimal=1)
    ax_stream.set_title(plot_title, y=title_y_adj)
    ax_stream.grid(zorder=0, c="lightgrey", alpha=0.5)
    #### streamplot ####

    #### colorbar ####
    cbax = fig.add_subplot(gridspec[0, 1])
    cb = Colorbar(
        ax=cbax,
        mappable=stream.lines,
        ticklocation="right",
    )
    cb.outline.set_visible(False)
    cb.set_label("Velocity", rotation=0, labelpad=25)
    
    # save and display plot
    if save_path:
        fig.savefig(save_path, bbox_inches='tight') 
    plt.show()