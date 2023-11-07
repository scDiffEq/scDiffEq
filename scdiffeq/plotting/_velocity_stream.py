

# -- import packages: ----------------------------------------------------------
import os
import abc
import scvelo as scv
import numpy as np
import anndata
import ABCParse


# -- import local dependencies: ------------------------------------------------
from ..core import utils


# -- set typing: ---------------------------------------------------------------
from typing import Union
NoneType = type(None)


# -- Helper classes: -----------------------------------------------------------
class AnnDataAttributeInspector(ABCParse.ABCParse):
    def __init__(self):
        ...

    @property
    @abc.abstractmethod
    def _KEY_SET(self):
        ...

    def transfer(self, old_adata: anndata.AnnData, new_adata: anndata.AnnData):

        for group, key_set in self._KEY_SET.items():
            for key in key_set:
                getattr(new_adata, group)[key] = getattr(old_adata, group)[key]


class NearestNeighborAnnDataInspector(AnnDataAttributeInspector):
    def __init__(
        self,
        obsp_neighbor_keys=["distances", "connectivities"],
        uns_neighbor_keys=["neighbors"],
    ):
        super()
        self.__parse__(locals(), public=[None])

    @property
    def _KEY_SET(self):
        return {"obsp": self._obsp_neighbor_keys, "uns": self._uns_neighbor_keys}

    @property
    def has_obsp_neighbor_keys(self):
        return np.all([key in self.adata.obsp for key in self._obsp_neighbor_keys])

    @property
    def has_uns_neighbor_keys(self):
        return np.all([key in self.adata.uns_keys() for key in self._uns_neighbor_keys])

    @property
    def has_nn(self):
        return np.all([self.has_obsp_neighbor_keys, self.has_uns_neighbor_keys])

    def __call__(self, adata):
        self.__update__(locals())
        return self.has_nn


class VelocityGraphInspector(AnnDataAttributeInspector):
    def __init__(self, velo_key="Z_drift", basis="umap"):
        super()
        self.__parse__(locals(), public=[None])

    @property
    def _KEY_SET(self):
        """_SCVELO_ADDED_GRAPH_KEYS"""
        return {
            "uns": [
                f"{self._velo_key}_graph",
                f"{self._velo_key}_graph_neg",
                f"{self._velo_key}_params",
            ],
            #             "obsm": [f"{self._velo_key}_{self._basis}"],
            "obs": [f"{self._velo_key}_self_transition"],
        }

    @property
    def _VELO_GRAPH_COMPUTED(self):

        _GROUPS = []
        for group, key_set in self._KEY_SET.items():
            _GROUPS.append(
                np.all([key in getattr(self.adata, group) for key in key_set])
            )
        return np.all(_GROUPS)

    def __call__(self, adata):
        self.__update__(locals())
        return self._VELO_GRAPH_COMPUTED
    
class VelocityAnnDataFormatter(ABCParse.ABCParse):
    """Reformatted, minimal adata for velo plots"""

    def __init__(
        self,
        state_key="X_pca",
        drift_layer_key="Z_drift",
        diffusion_layer_key="Z_diffusion",
        drift_obs_key="drift",
        diffusion_obs_key="diffusion",
    ):
        self.__parse__(locals(), public=[None])

        self._NN_INSPECTOR = NearestNeighborAnnDataInspector()
        self._VELO_GRAPH_INSPECTOR = VelocityGraphInspector()

    @property
    def Z_state(self):
        return self.adata.obsm[self._state_key]

    @property
    def X_umap(self):
        return self.adata.obsm["X_umap"]

    @property
    def Z_drift(self):
        return self.adata.obsm[self._drift_layer_key]

    @property
    def Z_diffusion(self):
        return self.adata.obsm[self._diffusion_layer_key]

    @property
    def obs_df(self):
        obs_df = self.adata.obs[[self._drift_obs_key, self._diffusion_obs_key]].copy()
        for col in obs_df.columns:
            obs_df[col] = obs_df[col].astype(float).values
        return obs_df

    @property
    def HAS_NN(self):
        return self._NN_INSPECTOR(self.adata)

    @property
    def HAS_VELOCITY_GRAPH(self):
        return self._VELO_GRAPH_INSPECTOR(self.adata)

    @property
    def adata_velo(self):
        if not hasattr(self, "_adata_velo"):
            self._adata_velo = anndata.AnnData(
                X=self.Z_state,
                dtype=self.Z_state.dtype,
                layers={
                    self._drift_layer_key: self.Z_drift,
                    self._diffusion_layer_key: self.Z_diffusion,
                    "spliced": self.Z_state,
                },
                obs=self.obs_df,
                obsm={"X_umap": self.X_umap},
            )
            if self._NN_INSPECTOR(self.adata):
                self._NN_INSPECTOR.transfer(self.adata, self._adata_velo)
            else:
                scv.pp.neighbors(self._adata_velo, use_rep="X")
        return self._adata_velo

    def __call__(self, adata: anndata.AnnData):

        self.__update__(locals())
        return self.adata_velo


# -- Main operational class: ---------------------------------------------------
class VelocityStreamPlot(ABCParse.ABCParse):
    _GRAPH_COMPUTED = []

    def __init__(
        self,
        state_key="X_pca",
        drift_layer_key="Z_drift",
        diffusion_layer_key="Z_diffusion",
        drift_obs_key="drift",
        diffusion_obs_key="diffusion",
    ):
        self.__parse__(locals(), public=["adata"])

        self.velocity_formatter = VelocityAnnDataFormatter(
            **utils.function_kwargs(VelocityAnnDataFormatter, self._PARAMS)
        )

    @property
    def adata_velo(self):
        if not hasattr(self, "_adata_velo"):
            self._adata_velo = self.velocity_formatter(self.adata)
        return self._adata_velo

    def _compute_velocity_graph(
        self, velocity_key="Z_drift", n_jobs=int(os.cpu_count() / 2), *args, **kwargs
    ):
        self._velo_key = velocity_key
        if self.velocity_formatter.HAS_VELOCITY_GRAPH:
            self.velocity_formatter._VELO_GRAPH_INSPECTOR.transfer(
                self.adata,
                self.adata_velo,
            )
        else:
            scv.tl.velocity_graph(
                self.adata_velo, vkey=velocity_key, n_jobs=n_jobs, **kwargs
            )
            self.velocity_formatter._VELO_GRAPH_INSPECTOR.transfer(
                self.adata_velo, self.adata
            )

        self._GRAPH_COMPUTED.append(velocity_key)

    @property
    def _VMIN(self):
        return np.quantile(self.adata_velo.obs[self.color], self.color_quantile)

    @property
    def _VMAX(self):
        return np.quantile(self.adata_velo.obs[self.color], (1 - self.color_quantile))

    def __call__(
        self,
        adata,
        velocity_key="Z_drift",
        color="diffusion",
        basis="umap",
        color_quantile=0.05,
        cmap="plasma",
        save: Union[bool, str] = False,
        **kwargs,
    ):

        self.__update__(locals())

        self._adata_velo = self.velocity_formatter(self.adata)

        if not velocity_key in self._GRAPH_COMPUTED:
            self._compute_velocity_graph(velocity_key=self.velocity_key)

        scv.pl.velocity_embedding_stream(
            self.adata_velo,
            vkey=self.velocity_key,
            basis=self.basis,
            color=self.color,
            cmap=self.cmap,
            vmin=self._VMIN,
            vmax=self._VMAX,
            save=self.save,
            **kwargs,
        )
        emb_key_added = f"{velocity_key}_{basis}"
        adata.obsm[emb_key_added] = self.adata_velo.obsm[emb_key_added]
        

# -- API-facing function: ------------------------------------------------------
def velocity_stream(
    adata: anndata.AnnData,
    velocity_key: str = "Z_drift",
    color: str = "diffusion",
    state_key: str = "X_pca",
    drift_layer_key: str = "Z_drift",
    diffusion_layer_key: str = "Z_diffusion",
    drift_obs_key: str = "drift",
    diffusion_obs_key: str = "diffusion",
    basis: str = "umap",
    color_quantile: float = 0.05,
    cmap: str ="plasma",
    save: bool = False,
    **kwargs,
):
    """
    Parameters:
    -----------
    adata
        type: anndata.AnnData
        default: [ REQUIRED ]
        
    velocity_key
        type: str
        default: "Z_drift"
        
    color
        type: str
        default: "diffusion"
        
    state_key
        type: str
        default: "X_pca"
        
    drift_layer_key
        type: str
        default: "Z_drift"
        
    diffusion_layer_key
        type: str
        default: "Z_diffusion"
        
    drift_obs_key
        type: str
        default: "drift"
        
    diffusion_obs_key
        type: str
        default: "diffusion"
        
    basis
        type: str
        default: "umap"
        
    color_quantile
        type: float
        default: 0.05
        
    cmap
        type: str
        default: "plasma"
        
    save
        type: bool
        default: False
    
    Returns:
    --------
    None, plots velocity stream plot.
    
    Notes:
    ------
    1.  Increased flexibility with passed parameters is a TODO, will
        be mostly through **kwargs.
    """
    velo_stream_plot = VelocityStreamPlot(
        state_key=state_key,
        drift_layer_key=drift_layer_key,
        diffusion_layer_key=diffusion_layer_key,
        drift_obs_key=drift_obs_key,
        diffusion_obs_key=diffusion_obs_key,
    )
    velo_stream_plot(
        adata=adata,
        velocity_key=velocity_key,
        color=color,
        basis=basis,
        color_quantile=color_quantile,
        cmap=cmap,
        save=save,
    )
    