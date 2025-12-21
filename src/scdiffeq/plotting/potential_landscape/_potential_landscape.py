
# -- import packages: ----------------------------------------------------------
import plotly.express as px
import plotly.graph_objects as go
import anndata
import ABCParse

# -- import local dependencies: ------------------------------------------------
from ...core import utils
from ._neighbor_smoothing import NeighborSmoothing
from ._surface_3d import Surface3D
from ._scatter_3d import Scatter3D


# -- set typing: ---------------------------------------------------------------
from typing import Union, Dict

NoneType = type(None)


# -- Operational class: --------------------------------------------------------
class UMAPSurfaceScatterPlot3D(ABCParse.ABCParse):
    """Manager class of both the surface and the scatter"""

    def __init__(
        self,
        use_key="X_umap",
        z_key="psi",
        x_label="UMAP-1",
        y_label="UMAP-2",
        z_label="Ψ, Cell Potential",
        scatter_color="Cell type annotation",
        smoothing_mode: Union[NoneType, str] = "radius",
        radius: int = 1,
        neighbors: int = 15,
        width: float = 800,
        height: float = 800,
        gridpoints: int = 250,
        surface_cmap: str = "aggrnyl",
        scatter_cmap: Union[NoneType, Dict] = None,
        marker_size: int = 2,
        surface_opacity: float = 0.8,
        showscale_surface: bool = False,
        show_surface: bool = True,
        show_scatter: bool = True,
        backgroundcolor = "white",
        gridcolor = "grey",
        showbackground = True,
        zerolinecolor = "white",
        **kwargs,
    ):
        self.__parse__(locals(), public=[None])

    def _fit_neighbor_model(self):
        """Neighbor model is fit to surface data"""
        self.neighbor_model = NeighborSmoothing(
            self._smoothing_mode, self._radius, self._neighbors
        )
        self.neighbor_model.fit(
            self.adata_surface.obsm[self._use_key],
            self.adata_surface.obs[self._z_key].values,
        )

    def _prepare_surface(self):
        surface = Surface3D()
        return surface(self.adata_surface, self.neighbor_model)

    def _prepare_scatter(self):
        scatter = Scatter3D()
        return scatter(self.adata_scatter, self.neighbor_model)

    @property
    def SURFACE(self):
        self.surface_df = self._prepare_surface()
        if not hasattr(self, "_surface"):
            self._surface = go.Surface(
                x=self.surface_df.index,
                y=self.surface_df.columns,
                z=self.surface_df.values,
                opacity=self._surface_opacity,
                colorscale=self._surface_cmap,
                showscale=self._showscale_surface,
            )
        return self._surface

    @property
    def _PLOTLY_SCATTER_KWARGS(self):
        _kw = {"x": "UMAP-1", "y": "UMAP-2", "z": self._z_key}
        if not isinstance(self._scatter_color, NoneType):
            _kw["color"] = self._scatter_color
            self.scatter_df[self._scatter_color] = self.adata_scatter.obs[
                self._scatter_color
            ].values
            if not isinstance(self._scatter_cmap, NoneType):
                _kw["color_discrete_map"] = self._scatter_cmap
        return _kw

    @property
    def SCATTER(self):
        self.scatter_df = self._prepare_scatter()
        if not hasattr(self, "_scatter"):
            self._scatter = px.scatter_3d(
                self.scatter_df, **self._PLOTLY_SCATTER_KWARGS
            )
            self._scatter.update_traces(marker=dict(size=self._marker_size))
        return self._scatter.data

    @property
    def _PLOT_DATA(self):
        _data_to_plot = []
        if self._show_surface:
            _data_to_plot += [self.SURFACE]
        if self._show_scatter:
            _data_to_plot += list(self.SCATTER)
        return _data_to_plot
    
    @property
    def _AXIS_DICT(self):
        return dict(
                    backgroundcolor=self._backgroundcolor,
                    gridcolor=self._gridcolor,
                    showbackground=self._showbackground,
                    zerolinecolor=self._zerolinecolor,
                )

    def plot(self):
        
        fig = go.Figure(data=self._PLOT_DATA)
        fig.update_layout(
            autosize=True,
            width=self._width,
            height=self._height,
            paper_bgcolor="white",
            legend=dict(font=dict(size=14), itemsizing="constant", y=0.2, x=1),
            scene=dict(
                xaxis_title=self._x_label,
                yaxis_title=self._y_label,
                zaxis_title=self._z_label,
                xaxis=self._AXIS_DICT,
                yaxis=self._AXIS_DICT,
                zaxis=self._AXIS_DICT,
            ),
        )
        fig.show()

    def __call__(self, adata_surface, adata_scatter):

        self.__update__(locals())
        self._fit_neighbor_model()
        self.plot()


# -- API-facing function: ------------------------------------------------------
def potential_landscape(
    adata_surface: anndata.AnnData,
    adata_scatter: anndata.AnnData,
    use_key: str = "X_umap",
    z_key: str = "psi",
    x_label: str = "UMAP-1",
    y_label: str = "UMAP-2",
    z_label: str = "Ψ, Cell Potential",
    scatter_color=None,
    smoothing_mode: Union[NoneType, str] = "radius",
    radius: int = 1,
    neighbors: int = 15,
    width: float = 800,
    height: float = 800,
    gridpoints=250,
    surface_cmap: str = "aggrnyl",
    scatter_cmap=None,
    marker_size=2,
    surface_opacity=0.8,
    showscale_surface: bool = False,
    show_surface: bool = True,
    show_scatter: bool = True,
    backgroundcolor = "white",
    gridcolor = "grey",
    showbackground = True,
    zerolinecolor = "white",
):
    """
    Plot the 3D potential landscape and simulated cells within the landscape.

    Parameters:
    -----------
    adata_surface
        type: anndata.AnnData
        default: None

    adata_scatter
        type: anndata.AnnData
        default: None

    use_key
        type: str
        default: "X_umap"

    z_key
        type: str
        default: "psi"

    x_label
        type: str
        default: "UMAP-1"

    y_label
        type: str
        default: "UMAP-2"

    z_label
        type: str
        default: "Ψ, Cell Potential"

    scatter_color
        type: [NoneType, str]
        default: None

    smoothing_mode
        type: [NoneType, str]
        default: "radius"

    radius
        type: float
        default: 1

    neighbors
        type: int
        default: 15

    width
        type: float
        default: 800

    height
        type: float
        default: 800

    gridpoints
        type: int
        default: 250

    surface_cmap
        type: str
        default: "aggrnyl"

    scatter_cmap
        type: [NoneType, Dict]
        default: None

    marker_size
        type: float
        default: 2.0

    surface_opacity
        type: float
        default: 0.8

    showscale_surface
        type: bool
        default: False

    show_surface
        type: bool
        default: True

    show_scatter
        type: bool
        default: True

    Returns:
    --------
    None, displays plotly figure.

    Notes:
    ------
    """
    landscape_plot = UMAPSurfaceScatterPlot3D(
        **utils.function_kwargs(UMAPSurfaceScatterPlot3D, locals())
    )
    landscape_plot(adata_surface, adata_scatter)