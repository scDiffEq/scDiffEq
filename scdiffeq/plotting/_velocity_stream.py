# -- import packages: ---------------------------------------------------------
import ABCParse
import anndata
import matplotlib.pyplot as plt
import numpy as np
import cellplots as cp
import matplotlib.cm
import os
import pathlib


# -- import local dependencies: -----------------------------------------------
from ..tools import VelocityEmbedding, GridVelocity


# -- type setting: ------------------------------------------------------------
from typing import Any, Dict, List, Optional, Union, Tuple


# -- Operational class: -------------------------------------------------------
class VelocityStreamPlot(ABCParse.ABCParse):
    """Velocity stream plot for a single ax"""

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
    ):
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
        """ """
        obs_df = self._adata.obs.copy().reset_index()
        cols = obs_df.columns.tolist()

        kwargs = self._SCATTER_KWARGS

        COLOR_FROM_OBS = self._c in cols

        if COLOR_FROM_OBS:
            COLOR_BY_GROUP = str(obs_df[self._c].dtype) == "categorical"
            if not COLOR_BY_GROUP: # implies float not grouped object.
                c_idx = np.argsort(obs_df[self._c])
                kwargs.update({"c":obs_df[self._c][c_idx]})
                self._img = ax.scatter(self.X_emb[c_idx, 0], self.X_emb[c_idx, 1], **kwargs)
                if not self._disable_cbar:
                    cbar = plt.colorbar(mappable = self._img, **self._cbar_kwargs)
                    cbar.solids.set(alpha=1)
#                     cbar.set_alpha(1)
                
            else:
                kwargs.pop("c")
                groups = obs_df.groupby(self._c).groups # dict
                cmap = self._SCATTER_CMAP(groups)
                for group, group_ix in groups.items():
                    if hasattr(self, "_group_zorder") and group in self._group_zorder:
                        kwargs.update({'zorder': self._group_zorder[group]})
                    ax.scatter(
                        self.X_emb[group_ix, 0], self.X_emb[group_ix, 1], color = cmap[group], **kwargs,
                    )
        else:
            ax.scatter(self.X_emb[:, 0], self.X_emb[:, 1], **kwargs)

    @property
    def scdiffeq_figure_dir(self):
        return pathlib.Path("scdiffeq_figures")

    def _mk_fig_dir(self):
        if not self.scdiffeq_figure_dir.exists():
            os.mkdir(self.scdiffeq_figure_dir)
            self._INFO(f"mkdir: {self.scdiffeq_figure_dir}")

    @property
    def sdq_info(self):
        return self._adata.uns["sdq_info"]

    @property
    def data_model_info_tag(self) -> str:
        return f"{self.sdq_info['project']}.version_{self.sdq_info['version']}.ckpt_{self.sdq_info['ckpt']}"

    @property
    def fname_basis(self):
        if "sdq_info" in self._adata.uns:
            try:
                return self.scdiffeq_figure_dir.joinpath(f"velocity_stream.{self.data_model_info_tag}")
            except:
                return self.scdiffeq_figure_dir.joinpath(f"velocity_stream.{self.sdq_info}")
            finally:
                pass
        return self.scdiffeq_figure_dir.joinpath("velocity_stream")
    
    @property
    def SVG_path(self):
        return pathlib.Path(".".join([str(self.fname_basis), "svg"]))
    
    @property
    def PNG_path(self):
        return pathlib.Path(".".join([str(self.fname_basis), "png"]))
    
    def save_img(self):
        """"""
        self._mk_fig_dir()
        plt.savefig(self.SVG_path, dpi = self._svg_dpi)
        plt.savefig(self.PNG_path, dpi = self._png_dpi)
        self._INFO(f"Saved to: \n  {self.SVG_path}\n  {self.PNG_path}")
        
    def __call__(
        self,
        adata: anndata.AnnData,
        ax: Optional[Union[plt.Axes, List[plt.Axes]]] = None,
        stream_color: str = "k",
        c: str = "dodgerblue",
        group_zorder: Optional[Dict] = None,
        cmap: Optional[Union[Dict,List,Tuple, str]] = 'plasma_r',
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
        """
        Args:

        Returns
            None
        """
        self.__update__(locals())
        
        if ax is None:
            _mpl_kwargs = {
                "nplots": 1,
                "ncols": 1,
                "height": 1.,
                "width": 1.,
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
    cmap: Optional[Union[Dict,List,Tuple]] = 'plasma_r',
    group_zorder: Optional[Dict] = None,
    linewidth: float = 0.5,
    stream_density: float = 2.5,
    add_margin: float = 0.1,
    arrowsize: float = 1,
    arrowstyle: str = "-|>",
    maxlength: float = 4,
    integration_direction: str = "both",
    scatter_zorder: int = 0,
    stream_zorder: int = 10,
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
    png_dpi: Optional[float] = 500,
    svg_dpi: Optional[float] = 250,
    *args,
    **kwargs,
):
    """"""

    init_kwargs = ABCParse.function_kwargs(VelocityStreamPlot.__init__, locals())
    call_kwargs = ABCParse.function_kwargs(VelocityStreamPlot.__call__, locals())
    velo_stream_plot = VelocityStreamPlot(**init_kwargs)
    axes = velo_stream_plot(**call_kwargs)
    
    if return_axes:
        return axes
