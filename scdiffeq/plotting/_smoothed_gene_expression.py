# -- import packages: ---------------------------------------------------------
import ABCParse
import anndata
import cellplots
import logging
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import pathlib
import scdiffeq_plots as sdq_pl

# -- import local dependencies: -----------------------------------------------
from ..core import utils

# -- set type hints: ----------------------------------------------------------
from typing import Dict, List, Optional, Union

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -- cls: ---------------------------------------------------------------------
class FillBetweenPlot:
    """KNOWN BUG: CELLPLOTS REQUIRED / NOT YET RELEASED"""

    def __init__(self, *args, **kwargs): ...

    @property
    def _MEAN(self):
        return self._df["mean"]

    @property
    def _STD(self):
        return self._df["std"]

    @property
    def X(self):
        return self._MEAN.index

    @property
    def Y(self):
        return self._MEAN

    @property
    def YMIN(self):
        return self._MEAN - self._STD

    @property
    def YMAX(self):
        return self._MEAN + self._STD

    def _configure_plot(self, ax):
        if ax is None:
            # KNOWN BUG
            self.fig, axes = cellplots.plot(1, 1, delete=[["top", "right"]])
            ax = axes[0]
        return ax

    def _plot(self, label=None, *args, **kwargs):
        self.ax.fill_between(x=self.X, y1=self.YMIN, y2=self.YMAX, alpha=0.2, zorder=0)
        self.ax.plot(self.X, self.Y, zorder=1, label=label)

    def __call__(
        self,
        df,
        ax: Optional[matplotlib.axes.Axes] = None,
        label: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        self._df = df
        self.ax = self._configure_plot(ax)
        self._plot(label, **kwargs)


class SmoothedGEXPlot(ABCParse.ABCParse):
    def __init__(
        self,
        smoothed_gex_key: str = "t_smoothed_gex",
        nplots=1,
        ncols=1,
        xy_spines=True,
        height=0.5,
        width=0.5,
        ticklabel_size=6,
        x_label: Optional[str] = None,
        y_label: str = "Z-score",
        label_fontsize: float = 8,
        legend_loc: tuple = (1, 0.4),
        legend_fontsize: float = 8,
        *args,
        **kwargs,
    ) -> None:
        self.__parse__(locals(), public=[None])

    @property
    def _PLOT_KWARGS(self):
        return utils.function_kwargs(func=sdq_pl.plot, kwargs=self._PARAMS)

    def _build_plot(self):
        self.fig, self.axes = sdq_pl.plot(**self._PLOT_KWARGS)

    def _formatting(self, ax):
        ax.tick_params(axis="both", which="both", labelsize=self._ticklabel_size)
        ax.set_xlabel(self._X_LABEL, fontsize=self._label_fontsize)
        ax.set_ylabel(self._y_label, fontsize=self._label_fontsize)
        ax.legend(
            facecolor="None",
            edgecolor="None",
            loc=self._legend_loc,
            fontsize=self._legend_fontsize,
        )

    @property
    def _SMOOTHED_GEX(self) -> Dict:
        return self.adata_sim.uns[self._smoothed_gex_key]

    def forward(self, gene, gene_df):

        fill_between_plot = FillBetweenPlot()
        fill_between_plot(gene_df, ax=self.axes[0], label=gene)

    @property
    def _X_LABEL(self):
        if self._x_label is None:
            return self._smoothed_gex_key.split("_")[0]
        else:
            return self._x_label

    @property
    def _SAVE_PATH(self):
        if self._save_path is None:
            return pathlib.Path(
                os.path.join(
                    os.getcwd(),
                    "scDiffEq_figures",
                    ".".join(
                        [
                            "smoothed_gex",
                            "_".join(list(self._SMOOTHED_GEX.keys())),
                            self._X_LABEL,
                            "svg",
                        ],
                    ),
                )
            )
        return self._save_path

    def __call__(
        self,
        adata_sim: anndata.AnnData,
        save: bool = False,
        save_path: Optional[Union[pathlib.Path, str]] = None,
        *args,
        **kwargs,
    ):

        self.__update__(locals(), private=["save"])
        self._save_path = save_path
        self._build_plot()

        for gene, gene_df in self._SMOOTHED_GEX.items():
            self.forward(gene, gene_df)

        self._formatting(ax=self.axes[0])

        if self._save:
            if not self._SAVE_PATH.parent.exists():
                self._SAVE_PATH.parent.mkdir()
                logger.info(f"Directory created: {self._SAVE_PATH.parent}")
            plt.savefig(self._SAVE_PATH)
            logger.info(f"Saved to: {self._SAVE_PATH}")


def plot_smoothed_expression(
    adata_sim: anndata.AnnData,
    smoothed_gex_key: str = "t_smoothed_gex",
    save: bool = False,
    save_path: Optional[Union[pathlib.Path, str]] = None,
    nplots=1,
    ncols=1,
    xy_spines=True,
    height=0.5,
    width=0.5,
    ticklabel_size=6,
    x_label: Optional[str] = None,
    y_label: str = "Z-score",
    label_fontsize: float = 8,
    legend_loc: tuple = (1, 0.4),
    legend_fontsize: float = 8,
    *args,
    **kwargs,
):
    """
    Parameters
    ----------
    adata_sim: anndata.AnnData

    smoothed_gex_key: str = "t_smoothed_gex",

    nplots=1,

    ncols=1,

    xy_spines=True,

    height=0.5,

    width=0.5,

    ticklabel_size=6,

    x_label: Optional[str] = None,

    y_label: str = "Z-score",

    label_fontsize: float = 8,

    legend_loc: tuple = (1, 0.4),

    legend_fontsize: float = 8,

    Returns
    -------

    Notes
    -----

    """
    pl_smooth_gex = SmoothedGEXPlot(**locals())
    pl_smooth_gex(adata_sim, save=save, save_path=save_path)
