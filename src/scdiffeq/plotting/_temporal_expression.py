# -- import packages: ---------------------------------------------------------
import anndata
import ABCParse
import matplotlib.pyplot as plt
import cellplots as cp
import matplotlib.cm as cm


# -- set typing: --------------------------------------------------------------
from typing import Optional, List, Dict, Union


# -- Operational class: -------------------------------------------------------
class TemporalExpressionPlot(ABCParse.ABCParse):
    def __init__(
        self,
        height: float = 0.4,
        width: float = 0.4,
        delete: List[List[str]] = [["top", "right"]],
        plot_kwargs: Dict = {},
        *args,
        **kwargs,
    ):
        self.__parse__(locals(), public=[None])

    @property
    def _MEAN(self):
        return self._adata_sim.uns["gex_mean"][self._gene]

    @property
    def _STD(self):
        return self._adata_sim.uns["gex_std"][self._gene]

    @property
    def _HI(self):
        return self._MEAN + self._STD

    @property
    def _LO(self):
        _lo = self._MEAN - self._STD
        _lo[_lo < 0] = 0
        return _lo

    @property
    def _columns(self):
        return self._MEAN.columns.tolist()

    @property
    def _X(self):
        return self._MEAN.index.to_numpy().round(2)

    @property
    def _global_max(self):
        return self._HI.max().max()

    @property
    def _PLOT_GROUPS(self):
        if not hasattr(self, "_plot_groups"):
            return self._columns
        return self._plot_groups

    @property
    def ax(self):
        if not hasattr(self, "_ax"):
            fig, axes = cp.plot(
                1,
                1,
                height=self._height,
                width=self._width,
                delete=self._delete,
                **self._plot_kwargs,
            )
            self._ax = axes[0]
        return self._ax

    @property
    def _CMAP(self):
        if not hasattr(self, "_cmap"):
            self._cmap = cm.tab20.colors
        return self._cmap

    def _fetch_color(self, en, group):

        if group in self._CMAP:
            return self._CMAP[group]
        return self._CMAP[en]

    def _plot_std(self, group, color):

        y1 = self._LO[group]
        y2 = self._HI[group]
        self.ax.fill_between(
            x=self._X,
            y1=y1,
            y2=y2,
            alpha=0.05,
            ec="None",
            linewidth=0,
            color=color,
        )

    @property
    def _sim_idx(self):
        return self._adata_sim.uns["idx"]

    def forward(self, en, group):

        color = self._fetch_color(en, group)

        if self._show_std:
            self._plot_std(group, color)

        x, y = self._X, self._MEAN[group]
        self.ax.plot(x, y, lw=2, color=color, zorder=en + 2, label=group)
        self.ax.set_xlim(x.min(), x.max())

    #         self.ax.set_ylim(0, round(self._global_max))

    def __call__(
        self,
        adata_sim: anndata.AnnData,
        gene: str,
        plot_groups: Optional[List[str]] = None,
        show_std: bool = False,
        ax: Optional[plt.Axes] = None,
        cmap: Optional[Dict] = None,
        save: bool = False,
        savename: str = None,
        save_format: str = "svg",
        dpi: int = 250,
        x_label: str = "t (d)",
        y_label: str = None,
        title: str = None,
        title_fontsize: int = 8,
        grid: bool = True,
        show_legend: bool = True,
        *args,
        **kwargs,
    ) -> None:
        self.__update__(locals(), public=[None])

        for en, group in enumerate(self._PLOT_GROUPS):
            self.forward(en, group)

        self.ax.set_xlabel(self._x_label, fontsize=8)

        if y_label is None:
            y_label = self._gene
        if title is None:
            title = f"sim idx: {self._sim_idx}"
        self.ax.set_ylabel(y_label, fontsize=8)
        self.ax.set_title(title, fontsize=title_fontsize)

        if grid:
            self.ax.grid(zorder=0, alpha=0.1)

        if show_legend:
            self.ax.legend(
                edgecolor="None", facecolor="None", loc=(1.05, 0.25), fontsize=6
            )

        if save:
            if savename is None:
                savename = f"scDiffEq.temporal_expression.sim_idx_{self._sim_idx}.{gene}.{save_format}"
            plt.savefig(savename, dpi=dpi)


# -- API-facing function: -----------------------------------------------------
def temporal_expression(
    adata_sim: anndata.AnnData,
    gene: str,
    plot_groups: Optional[List[str]] = None,
    show_std: bool = False,
    ax: Optional[plt.Axes] = None,
    cmap: Optional[Dict] = None,
    height: float = 0.4,
    width: float = 0.4,
    delete: List[List[str]] = [["top", "right"]],
    save: bool = False,
    savename: str = None,
    save_format: str = "svg",
    dpi: int = 250,
    x_label: str = "t (d)",
    y_label: str = None,
    title: str = None,
    grid: bool = True,
    show_legend: bool = True,
    plot_kwargs: Optional[Dict] = {},
    title_fontsize: int = 8,
):
    """
    Plot smoothed expression over time.

    Parameters
    ----------
    adata_sim: anndata.AnnData

    gene: str

    plot_groups: Optional[List[str]] = None

    show_std: bool = False

    ax: Optional[plt.Axes] = None

    cmap: Optional[Dict] = None

    height: float, default = 0.4

    width: float, default = 0.4

    delete: List[List[str]], default = [["top", "right"]]

    save: bool, default = False

    savename: str, default = None

    save_format: str, default = "svg"
        ["svg", "png"]

    dpi: int, default = 250

    plot_kwargs: Optional[Dict], default = {}

    Returns
    -------
    None
    """

    time_expr = TemporalExpressionPlot(
        height=height, width=width, delete=delete, plot_kwargs=plot_kwargs
    )
    time_expr(
        adata_sim=adata_sim,
        gene=gene,
        plot_groups=plot_groups,
        show_std=show_std,
        ax=ax,
        cmap=cmap,
        save=save,
        savename=savename,
        save_format=save_format,
        dpi=dpi,
        x_label=x_label,
        y_label=y_label,
        title=title,
        grid=grid,
        show_legend=show_legend,
        title_fontsize=title_fontsize,
    )
