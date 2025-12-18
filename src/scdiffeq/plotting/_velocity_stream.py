# -- import packages: ---------------------------------------------------------
import ABCParse
import anndata
import cellplots as cp
import logging
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

# -- import local dependencies: -----------------------------------------------
from ..tools import VelocityEmbedding, GridVelocity

# -- set type hints: ----------------------------------------------------------
from typing import Any, Dict, List, Optional, Union, Tuple

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -- Operational class: -------------------------------------------------------
class VelocityStreamPlot(ABCParse.ABCParse):
    """A class to generate and plot velocity stream plots on a given plt.Axes object.

    This class is designed to visualize the flow of cells (or particles) in a
    velocity field, commonly used in single-cell data analysis to represent cell
    trajectories over a reduced dimensionality space. It extends the ABCParse class
    to leverage its parsing capabilities for initializing and configuring the plot
    parameters.

    Attributes:
        density (float): The density of the grid for velocity vectors. Defaults to 1.
        smooth (float): Smoothing factor applied to the velocity field. Defaults to 0.5.
        n_neighbors (Optional[int]): Number of neighbors to consider for local averaging. Defaults to None.
        min_mass (float): Minimum mass (weight) threshold for considering a point in the velocity field. Defaults to 1.
        autoscale (bool): Flag to automatically scale the vectors. Defaults to True.
        stream_adjust (bool): Adjust the streamplot parameters for optimal visualization. Defaults to True.
        cutoff_percentile (float): Percentile for cutoff to ignore outlier velocities. Defaults to 0.05.
        velocity_key (str): Key in `adata` to access velocity vectors. Defaults to "velocity".
        self_transitions (bool): Whether to consider self-transitions in the velocity calculations. Defaults to True.
        use_negative_cosines (bool): Flag to use negative cosines to adjust directionality. Defaults to True.
        T_scale (float): Scaling factor for the transition matrix. Defaults to 10.
        args: Variable length argument list.
        kwargs: Arbitrary keyword arguments.

    Note:
        This class requires an AnnData object `adata` to be passed at call time, not at initialization.
    """

    def __init__(
        self,
        density: float = 1,
        smooth: float = 0.5,
        n_neighbors: Optional[int] = None,
        min_mass: float = 1,
        autoscale: bool = True,
        stream_adjust: bool = True,
        cutoff_percentile: float = 0.05,
        velocity_key: str = "velocity",
        self_transitions: bool = True,
        use_negative_cosines: bool = True,
        T_scale: float = 10,
        *args,
        **kwargs,
    ) -> None:
        """Initializes the VelocityStreamPlot object with parameters to configure the velocity stream plot.

        This method sets up the necessary parameters for generating a velocity stream plot, including the setup for velocity embedding and grid velocity calculation. It parses the arguments and initializes internal states needed for plotting.

        Args:
            density (float): Density of the grid for velocity vectors. Higher values create a denser grid. Defaults to 1.
            smooth (float): Smoothing factor applied to the velocity vectors, influencing the smoothness of the stream plot. Defaults to 0.5.
            n_neighbors (Optional[int]): Number of nearest neighbors to use for local averaging of velocities. If None, a default heuristic is used. Defaults to None.
            min_mass (float): Minimum mass (weight) threshold for considering a point in the velocity field. Helps in filtering out noise. Defaults to 1.
            autoscale (bool): If True, scales the magnitude of velocity vectors automatically based on the density and size of the plot. Defaults to True.
            stream_adjust (bool): If True, adjusts stream plot parameters for optimal visualization. Defaults to True.
            cutoff_percentile (float): Percentile for cutoff to filter out outlier velocities, specified as a fraction between 0 and 1. Defaults to 0.05.
            velocity_key (str): Key in the AnnData object `adata` to access velocity vectors. Defaults to "velocity".
            self_transitions (bool): If True, considers self-transitions in velocity calculations, affecting the direction and magnitude of vectors. Defaults to True.
            use_negative_cosines (bool): If True, uses negative cosines to adjust the directionality of vectors, potentially improving visualization clarity. Defaults to True.
            T_scale (float): Scaling factor for the transition matrix T, affecting the overall magnitude of velocity vectors. Defaults to 10.
            args: Additional positional arguments not specifically defined.
            kwargs: Additional keyword arguments not specifically defined.

        Note:
            The `__init__` method does not require the AnnData object `adata`. Instead, `adata` should be passed to the `__call__` method when generating the plots.
        """

        self.__parse__(locals())

        self._velocity_emb = VelocityEmbedding(
            velocity_key=velocity_key,
            self_transitions=self_transitions,
            use_negative_cosines=use_negative_cosines,
            T_scale=T_scale,
        )
        self._grid_velocity = GridVelocity(
            density=density,
            smooth=smooth,
            n_neighbors=n_neighbors,
            min_mass=min_mass,
            autoscale=autoscale,
            stream_adjust=stream_adjust,
            cutoff_percentile=cutoff_percentile,
        )

    @property
    def X_emb(self):
        if not hasattr(self, "_X_emb"):
            self._X_emb, self._V_emb = self._velocity_emb(self._adata)
        return self._X_emb

    @property
    def V_emb(self):
        if not hasattr(self, "_V_emb"):
            self._X_emb, self._V_emb = self._velocity_emb(self._adata)
        return self._V_emb

    @property
    def X_grid(self):
        if not hasattr(self, "_X_grid"):
            self._X_grid, self._V_grid = self._grid_velocity(self.X_emb, self.V_emb)
        return self._X_grid

    @property
    def V_grid(self):
        if not hasattr(self, "_V_grid"):
            self._X_grid, self._V_grid = self._grid_velocity(self.X_emb, self.V_emb)
        return self._V_grid

    @property
    def x(self):
        return self.X_grid[0]

    @property
    def y(self):
        return self.X_grid[1]

    @property
    def u(self):
        return self.V_grid[0]

    @property
    def v(self):
        return self.V_grid[1]

    @property
    def xmin(self):
        return np.min(self.X_emb[:, 0])

    @property
    def xmax(self):
        return np.max(self.X_emb[:, 0])

    @property
    def ymin(self):
        return np.min(self.X_emb[:, 1])

    @property
    def ymax(self):
        return np.max(self.X_emb[:, 1])

    @property
    def xmargin(self):
        return (self.xmax - self.xmin) * self._add_margin

    @property
    def ymargin(self):
        return (self.ymax - self.ymin) * self._add_margin

    def _set_margin(self, ax):
        """"""
        ax.set_xlim(self.xmin - self.xmargin, self.xmax + self.xmargin)
        ax.set_ylim(self.ymin - self.ymargin, self.ymax + self.ymargin)

    @property
    def _STREAMPLOT_KWARGS(self) -> Dict[str, Any]:
        kwargs = {
            "color": self._stream_color,
            "density": self._stream_density,
            "linewidth": self._linewidth,
            "zorder": self._stream_zorder,
            "arrowsize": self._arrowsize,
            "arrowstyle": self._arrowstyle,
            "maxlength": self._maxlength,
            "integration_direction": self._integration_direction,
        }
        kwargs.update(self._stream_kwargs)
        return kwargs

    def streamplot(self, ax) -> None:
        ax.streamplot(self.x, self.y, self.u, self.v, **self._STREAMPLOT_KWARGS)
        self._set_margin(ax)

    @property
    def _SCATTER_KWARGS(self) -> Dict[str, Any]:
        """ """
        kwargs = {
            "c": self._c,
            "zorder": self._scatter_zorder,
            "ec": "None",
            "alpha": 0.2,
            "s": 50,
            "cmap": self._cmap,
            "rasterized": self._rasterized,
        }
        kwargs.update(self._scatter_kwargs)
        return kwargs

    def _SCATTER_CMAP(self, groups) -> Dict:
        """ """
        if not hasattr(self, "_cmap"):
            self._cmap = matplotlib.cm.tab20.colors
        if not isinstance(self._cmap, Dict):
            self._cmap = {group: self._cmap[en] for en, group in enumerate(groups)}
        return self._cmap

    def scatter(self, ax) -> None:
        """Generates a scatter plot on the given matplotlib axis, overlaying the stream plot.

        This method visualizes individual points (cells) on the velocity stream plot,
        with optional coloring and grouping.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which to plot the scatter plot.
        """
        obs_df = self._adata.obs.copy().reset_index()
        cols = obs_df.columns.tolist()

        kwargs = self._SCATTER_KWARGS

        COLOR_FROM_OBS = self._c in cols

        if COLOR_FROM_OBS:
            COLOR_BY_GROUP = str(obs_df[self._c].dtype) == "categorical"
            if not COLOR_BY_GROUP:  # implies float not grouped object.
                c_idx = np.argsort(obs_df[self._c])
                kwargs.update({"c": obs_df[self._c][c_idx]})
                self._img = ax.scatter(
                    self.X_emb[c_idx, 0], self.X_emb[c_idx, 1], **kwargs
                )
                if not self._disable_cbar:
                    cbar = plt.colorbar(mappable=self._img, **self._cbar_kwargs)
                    cbar.solids.set(alpha=1)
            #                     cbar.set_alpha(1)

            else:
                kwargs.pop("c")
                groups = obs_df.groupby(self._c).groups  # dict
                cmap = self._SCATTER_CMAP(groups)
                for group, group_ix in groups.items():
                    if hasattr(self, "_group_zorder") and group in self._group_zorder:
                        kwargs.update({"zorder": self._group_zorder[group]})
                    ax.scatter(
                        self.X_emb[group_ix, 0],
                        self.X_emb[group_ix, 1],
                        color=cmap[group],
                        **kwargs,
                    )
        else:
            ax.scatter(self.X_emb[:, 0], self.X_emb[:, 1], **kwargs)

    @property
    def scdiffeq_figure_dir(self):
        return pathlib.Path("scdiffeq_figures")

    def _mk_fig_dir(self):
        if not self.scdiffeq_figure_dir.exists():
            os.mkdir(self.scdiffeq_figure_dir)
            logger.info(f"mkdir: {self.scdiffeq_figure_dir}")

    @property
    def sdq_info(self):
        return self._adata.uns["sdq_info"]

    @property
    def data_model_info_tag(self) -> str:
        return f"{self.sdq_info['project']}.version_{self.sdq_info['version']}.ckpt_{self.sdq_info['ckpt']}"

    @property
    def fname_basis(self) -> pathlib.Path:
        if "sdq_info" in self._adata.uns:
            try:
                return self.scdiffeq_figure_dir.joinpath(
                    f"velocity_stream.{self.data_model_info_tag}"
                )
            except:
                return self.scdiffeq_figure_dir.joinpath(
                    f"velocity_stream.{self.sdq_info}"
                )
            finally:
                pass
        return self.scdiffeq_figure_dir.joinpath("velocity_stream")

    @property
    def SVG_path(self) -> pathlib.Path:
        """ """
        return pathlib.Path(".".join([str(self.fname_basis), "svg"]))

    @property
    def PNG_path(self) -> pathlib.Path:
        return pathlib.Path(".".join([str(self.fname_basis), "png"]))

    def save_img(self) -> None:
        """Saves the generated plot to both SVG and PNG formats in a specified directory."""
        self._mk_fig_dir()
        plt.savefig(self.SVG_path, dpi=self._svg_dpi)
        plt.savefig(self.PNG_path, dpi=self._png_dpi)
        logger.info(f"Saved to: \n  {self.SVG_path}\n  {self.PNG_path}")

    def __call__(
        self,
        adata: anndata.AnnData,
        ax: Optional[Union[plt.Axes, List[plt.Axes]]] = None,
        stream_color: str = "k",
        c: str = "dodgerblue",
        group_zorder: Optional[Dict] = None,
        cmap: Optional[Union[Dict, List, Tuple, str]] = "plasma_r",
        linewidth: float = 0.5,
        stream_density: float = 2.5,
        add_margin: float = 0.1,
        arrowsize: float = 1,
        density: float = 1,
        arrowstyle: str = "-|>",
        maxlength: float = 4,
        integration_direction: str = "both",
        scatter_zorder: int = 0,
        stream_zorder: int = 10,
        rasterized: bool = True,
        mpl_kwargs: Optional[Dict] = {},
        scatter_kwargs: Optional[Dict] = {},
        stream_kwargs: Optional[Dict] = {},
        cbar_kwargs: Optional[Dict] = {},
        disable_scatter: bool = False,
        disable_cbar: bool = False,
        save: bool = False,
        png_dpi: Optional[float] = 500,
        svg_dpi: Optional[float] = 250,
        *args,
        **kwargs,
    ):
        """Generates velocity stream plots for the provided AnnData object.

        Args:
            adata (anndata.AnnData): The AnnData object containing the data for plotting.
            ax (Optional[Union[plt.Axes, List[plt.Axes]]]): A matplotlib axis or list of axes where plots will be drawn.
            **kwargs: Additional keyword arguments to customize the plot appearance.

        Returns:
            List[plt.Axes]: A list of matplotlib axes with the generated plots.
        """
        self.__update__(locals())

        if ax is None:
            _mpl_kwargs = {
                "nplots": 1,
                "ncols": 1,
                "height": 1.0,
                "width": 1.0,
                "delete": "all",
                "del_xy_ticks": [True],
            }
            _mpl_kwargs.update(mpl_kwargs)
            fig, axes = cp.plot(**_mpl_kwargs)

        else:
            axes = ABCParse.as_list(ax)

        for ax in axes:
            self.streamplot(ax)
            if not self._disable_scatter:
                self.scatter(ax)

        if self._save:
            self.save_img()

        return axes


# -- API-facing function: -----------------------------------------------------
def velocity_stream(
    adata: anndata.AnnData,
    ax: Optional[Union[plt.Axes, List[plt.Axes]]] = None,
    c: str = "dodgerblue",
    cmap: Optional[Union[Dict, List, Tuple]] = "plasma_r",
    group_zorder: Optional[Dict] = None,
    linewidth: float = 0.5,
    stream_density: float = 2.5,
    add_margin: float = 0.1,
    arrowsize: float = 1,
    arrowstyle: str = "-|>",
    maxlength: float = 4,
    integration_direction: str = "both",
    scatter_zorder: int = 101,
    stream_zorder: int = 201,
    density: float = 1,
    smooth: float = 0.5,
    n_neighbors: Optional[int] = None,
    min_mass: float = 1,
    autoscale=True,
    stream_adjust=True,
    cutoff_percentile: float = 0.05,
    velocity_key: str = "velocity",
    self_transitions: bool = True,
    use_negative_cosines: bool = True,
    T_scale: float = 10,
    disable_scatter: bool = False,
    disable_cbar: bool = False,
    stream_kwargs: Optional[Dict[str, Any]] = {},
    scatter_kwargs: Optional[Dict[str, Any]] = {},
    cbar_kwargs: Optional[Dict] = {},
    mpl_kwargs: Optional[Dict[str, Any]] = {},
    return_axes: bool = False,
    save: Optional[bool] = False,
    rasterized: bool = True,
    png_dpi: Optional[float] = 500,
    svg_dpi: Optional[float] = 250,
    *args,
    **kwargs,
) -> Optional[Union[List[plt.Axes], None]]:
    """
    Generates velocity stream plots for single-cell data using the
    VelocityStreamPlot class.

    This function is a convenient wrapper around the VelocityStreamPlot
    class, allowing users to quickly generate and customize velocity stream
    plots without manually instantiating the class.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the data to plot.
    ax : Optional[Union[plt.Axes, List[plt.Axes]]], optional
        Matplotlib axes object or list of axes objects on which to draw the plots. If None, a new figure and axes are created. **Default**: ``None``.
    c : str, optional
        Color for the scatter plot points. Can be a column name from `adata.obs` if coloring by a categorical variable. **Default**: ``"dodgerblue"``.
    cmap : Optional[Union[Dict, List, Tuple, str]], optional
        Colormap for the scatter plot points if `c` is a categorical variable. **Default**: ``"plasma_r"``.
    group_zorder : Optional[Dict], optional
        Z-order for groups in the scatter plot, allowing certain groups to be plotted on top of others. **Default**: ``None``.
    linewidth : float, optional
        Line width for the streamlines. **Default**: ``0.5``.
    stream_density : float, optional
        Density of the streamlines. Higher values create more densely packed streamlines. **Default**: ``2.5``.
    add_margin : float, optional
        Additional margin added around the plotted data, specified as a fraction of the data range. **Default**: ``0.1``.
    arrowsize : float, optional
        Size of the arrows in the stream plot. **Default**: ``1``.
    arrowstyle : str, optional
        Style of the arrows in the stream plot. **Default**: ``"-|>"``.
    maxlength : float, optional
        Maximum length of the arrows in the stream plot. **Default**: ``4``.
    integration_direction : str, optional
        Direction of integration for the streamlines, can be "forward", "backward", or "both". **Default**: ``"both"``.
    scatter_zorder : int, optional
        Z-order for scatter plot points, determining their layering. **Default**: ``101``.
    stream_zorder : int, optional
        Z-order for the streamlines, determining their layering. **Default**: ``201``.
    density : float, optional
        **Default**: ``1``.
    smooth : float, optional
        **Default**: ``0.5``.
    n_neighbors : Optional[int], optional
        **Default**: ``None``.
    min_mass : float, optional
        **Default**: ``1``.
    autoscale : bool, optional
        **Default**: ``True``.
    stream_adjust : bool, optional
        **Default**: ``True``.
    cutoff_percentile : float, optional
    velocity_key : str, optional
    self_transitions : bool, optional
    use_negative_cosines : bool, optional
    T_scale : float, optional
    disable_scatter : bool, optional
        If True, disables the scatter plot overlay on the stream plot. **Default**: ``False``.
    disable_cbar : bool, optional
        If True, disables the color bar for the scatter plot. Useful when `c` is numeric. **Default**: ``False``.
    stream_kwargs : Optional[Dict[str, Any]], optional
    scatter_kwargs : Optional[Dict[str, Any]], optional
    cbar_kwargs : Optional[Dict], optional
    mpl_kwargs : Optional[Dict[str, Any]], optional
        Additional keyword arguments for customizing the stream plot, scatter plot, color bar, and matplotlib figure, respectively.
    return_axes : bool, optional
        If True, returns the matplotlib axes with the generated plots. **Default**: ``False``.
    save : bool, optional
        If True, saves the generated plot to SVG and PNG formats. **Default**: ``False``.
    png_dpi : Optional[float], optional
        DPI settings for saving PNG images. **Default**: ``500``.
    svg_dpi : Optional[float], optional
        DPI settings for saving SVG images. **Default**: ``250``.

    Returns
    -------
    Optional[Union[List[plt.Axes], None]]
        A list of matplotlib axes with the generated plots, if ``return_axes == True``. Otherwise, returns ``None``.
    """

    init_kwargs = ABCParse.function_kwargs(VelocityStreamPlot.__init__, locals())
    call_kwargs = ABCParse.function_kwargs(VelocityStreamPlot.__call__, locals())
    velo_stream_plot = VelocityStreamPlot(**init_kwargs)
    axes = velo_stream_plot(**call_kwargs)

    if return_axes:
        return axes
