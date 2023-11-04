
# -- import packages: ----------------------------------------------------------
import torch
import anndata
import autodevice
import adata_query
import numpy as np
import pandas as pd
import time
import os
import ABCParse


# -- import local dependencies: ------------------------------------------------
from ..core import utils

from ._final_state_per_simulation import FinalStatePerSimulation
from ._cell_potential import normalize_cell_potential
from ._norm import L2Norm
from ._knn import kNN


# -- set typing: ---------------------------------------------------------------
from typing import Union, List, Optional
NoneType = type(None)


class Simulator(ABCParse.ABCParse):
    """Base class for the simulator containing the core functions."""
    def __init__(
        self,
        adata: Union[anndata.AnnData, NoneType] = None,
        DiffEq = None,
        idx = None,
        use_key: str = "X_pca",
        UMAP = None,
        PCA = None,
        t_min: Union[float, NoneType] = None,
        t_max: Union[float, NoneType] = None,
        dt: float = 0.1,
        N: int = 2000,
        device: Union[torch.device, str] = "cuda:0",
        time_key: str = "Time point",
        ref_cell_type_key="Cell type annotation",
        gene_ids_key: str = "gene_ids",
        simulation_key_added: str = "simulation",
        final_state_key_added: str = "final_state",
        normalize_potential: bool = True,
        potential_normalization_kwargs={},
        time_key_added: str = "t",
        silent: bool = False,
        graph = None,
        ref_kNN_key: str = "X_pca",
        obs_mapping_keys: List[str] = ["leiden", "Cell type annotation"],
        fate_key: str = "Cell type annotation",
        return_adata: bool = True,
        name: Union[str, NoneType] = None,
        save_h5ad: bool = False,
        gpu: bool = True,
        wd = ".",
        *args,
        **kwargs,
    ):
        """ 
        adata: Union[anndata.AnnData, NoneType] = None,
        DiffEq = None,
        idx = None,
        use_key: str = "X_pca",
        UMAP = None,
        PCA = None,
        t_min: Union[float, NoneType] = None,
        t_max: Union[float, NoneType] = None,
        dt: float = 0.1,
        N: int = 2000,
        device: Union[torch.device, str] = "cuda:0",
        time_key: str = "Time point",
        ref_cell_type_key="Cell type annotation",
        gene_ids_key: str = "gene_ids",
        simulation_key_added: str = "simulation",
        final_state_key_added: str = "final_state",
        normalize_potential: bool = True,
        potential_normalization_kwargs={},
        time_key_added: str = "t",
        silent: bool = False,
        graph = None,
        ref_kNN_key: str = "X_pca",
        obs_mapping_keys: List[str] = ["leiden", "Cell type annotation"],
        fate_key: str = "Cell type annotation",
        return_adata: bool = True,
        name: Union[str, NoneType] = None,
        save_h5ad: bool = False,
        wd = ".",
        """

        self.__parse__(locals(), public=['DiffEq', 'idx'], ignore=["adata"])
        self._INFO = utils.InfoMessage(silent=silent)
        self._L2Norm = L2Norm()
        self._adata_input = adata.copy()

    @property
    def kNN(self):
        if isinstance(self._graph, NoneType):
            self._graph = kNN(self._adata_input, use_key=self._ref_kNN_key)
        return self._graph

    @property
    def _T_MIN(self):
        if isinstance(self._t_min, NoneType):
            self._t_min = self._adata_input.obs[self._time_key].min()
        return self._t_min

    @property
    def _T_MAX(self):
        if isinstance(self._t_max, NoneType):
            self._t_max = self._adata_input.obs[self._time_key].max()
        return self._t_max

    @property
    def _T_RANGE(self):
        return self._T_MAX - self._T_MIN

    @property
    def _N_STEPS(self):
        return int(self._T_RANGE / self._dt + 1)

    @property
    def t(self):
        return torch.linspace(self._T_MIN, self._T_MAX, self._N_STEPS)

    @property
    def Z0(self):
        return adata_query.fetch(
            adata = self._adata_input[self.idx],
            key=self._use_key,
            torch = self._gpu,
            device=self._device,
        ).expand(self._N, -1)
    
    @property
    def _LATENT_DIMS(self):
        return self.Z0.shape[-1]

    @property
    def _TIME(self) -> np.ndarray:
        """time for adata"""
        return np.repeat(np.linspace(self._T_MIN, self._T_MAX, self._N_STEPS), self._N)

    @property
    def _POTENTIAL_MODEL(self):
        return self.DiffEq.DiffEq.mu_potential

    @property
    def X(self):
        return self.Z_hat.flatten(0, 1).numpy()

    def _compose_core_adata(self):
        """
        Notes:
        ------
        Simulated matrix no longer sparse - no sense saving as sparse matrix.
        """
        self._INFO("Formatting as AnnData")
        self.adata = anndata.AnnData(self.X)
        self.adata.obs["t"] = self._TIME
        self.adata.uns["sim_idx"] = self.idx
        self.adata.uns[self._gene_ids_key] = self._adata_input.var[self._gene_ids_key]

    def forward(self):
        n_nonzero_steps = int(self._N_STEPS - 1)
        self._INFO(f"Simulating {self._N} trajectories over {n_nonzero_steps} steps.")
        self.Z_hat = self.DiffEq(Z0=self.Z0, t=self.t).detach().cpu()
        self.Z_input = self.Z_hat.flatten(0, 1).to(self._device)
        self._compose_core_adata()

    def compute_drift(self):
        self._INFO("Computing per-cell drift")
        self.X_drift = self.DiffEq.DiffEq.drift(self.Z_input).detach().cpu()
        self.adata.obsm["X_drift"] = self.X_drift
        self.adata.obs["drift"] = self._L2Norm(self.X_drift)

    def compute_diffusion(self):
        self._INFO("Computing per-cell diffusion")
        self.X_diffusion = (
            self.DiffEq.DiffEq.diffusion(self.Z_input).detach().cpu().squeeze(-1)
        )
        self.adata.obsm["X_diffusion"] = self.X_diffusion
        self.adata.obs["diffusion"] = self._L2Norm(self.X_diffusion)

    def compute_potential(self):

        self._INFO("Computing per-cell potential")
        self.cell_potential = (
            self.DiffEq.DiffEq.mu(self.Z_input).detach().cpu().numpy().flatten()
        )
        self.adata.obs["_psi"] = self.cell_potential

        if self._normalize_potential:
            
            potential_normalization_kws={
                "q": 0.05,
                "kNN_use_key": "X",
                "knn_smoothing_iters": 5,
                "use_tqdm": False,
            }
            potential_normalization_kws.update(self._potential_normalization_kwargs)
            
            self._INFO("Normalizing per-cell potential")
            normalize_cell_potential(
                self.adata,
                use_key="_psi",
                key_added="psi",
                **potential_normalization_kws,
            )

    def map_obs(self):
        for obs_key in self._obs_mapping_keys:
            self._INFO(f"Mapping observed label: `{obs_key}` to simulated cells")
            mapped = self.kNN.aggregate(self.adata.X, obs_key=obs_key, max_only=True)
            mapped.index = mapped.index.astype(str)
            self.adata.obs = pd.concat([self.adata.obs.copy(), mapped], axis=1)

    def annotate_final_state(self):

        for obs_key in self._obs_mapping_keys:
            final_state = FinalStatePerSimulation(
                adata=self.adata,
                obs_key=obs_key,
                N_sim=self._N,
                time_key=self._time_key_added,
                sim_key=self._simulation_key_added,
            )
            self._INFO(f"Annotating final state per simulation: `{obs_key}`")
            final_state.annotate_final_state(key_added=f"final_state.{obs_key}")
            

    def count_fates(self, fate_key):
        """
        Parameters:
        -----------
        fate_key
        """
        df = self.adata.obs.copy()
        return df.loc[df[self._time_key_added] == self._T_MAX][fate_key].value_counts()

    @property
    def fate_counts(self):
        return self.count_fates(self._fate_key)
    
    def run_umap(self):
        if not isinstance(self._UMAP, NoneType):
            self._INFO("Projecting simulated cells into UMAP space")
            self._X_umap = self._UMAP.transform(self.X)
            self.adata.obsm["X_umap"] = self._X_umap

    def run_inverse_pca(self):
        if not isinstance(self._PCA, NoneType):
            self._INFO("Inverting PCA to feature space")
            self._X_gene = self._PCA.inverse_transform(self.X)
            self.adata.obsm["X_gene"] = self._X_gene

    @property
    def X_umap(self):
        if not hasattr(self, "_X_umap"):
            self.run_umap()
        return self._X_umap

    @property
    def X_gene(self):
        if not hasattr(self, "_X_gene"):
            self.run_inverse_pca()
        return self._X_gene

    @property
    def _H5AD_PATH(self):
        
        if not os.path.exists(self._wd):
            os.mkdir(self._wd)
            
        if not isinstance(self._name, NoneType):
            ID = ".".join([self._name, self.idx])
        else:
            ID = self.idx
        fname = f"adata.scDiffEq_simulated.{ID}.h5ad"
        return os.path.join(self._wd, fname)
    
    @property
    def _COMPUTE_TIME(self):
        if not hasattr(self, "_compute_time"):
            self._T_FINAL = time.time()
            self._compute_time = self._T_FINAL - self._T_INIT
        self._INFO("Compute time: {:.2f}".format(self._compute_time))

    def __call__(
        self,
        adata: Union[anndata.AnnData, NoneType] = None,
        DiffEq = None,
        idx: Union[str, int] = None,
        obs_mapping_keys: List[str] = ["leiden"],
        fate_key: str = "Cell type annotation",
        use_key: str = "X_pca",
        return_adata: bool = True,
        normalize_potential: bool = True,
        potential_normalization_kwargs={},
        name: Union[str, NoneType] = None,
        save_h5ad: bool = False,
        gene_ids_key: str = "gene_ids",
    ) -> anndata.AnnData:
        
        """Runs everything, if given."""

        # -- inputs: -----------------------------------------------------------
        self.__update__(
            locals(),
            private=[
                "use_key",
                "return_adata",
                "normalize_potential",
                "obs_mapping_keys",
                "potential_normalization_kwargs",
                "name",
                "gene_ids_key",
                "fate_key",
                "save_h5ad",
            ],
        )

        self._T_INIT = time.time()

        self._adata_input = adata.copy()

        # -- run ---------------------------------------------------------------
        self.forward()
        self.compute_drift()
        self.compute_diffusion()
        self.compute_potential()
        self.map_obs()
        self.annotate_final_state()
        self.run_inverse_pca()
        self.run_umap()

        # -- outputs: ----------------------------------------------------------

        if self._save_h5ad:
            self.adata.write_h5ad(self._H5AD_PATH)

        if self._return_adata:
            return self.adata

    def __repr__(self):
        return "scDiffEq Simulator"


def simulate(
    adata: anndata.AnnData,
    model,
    idx,
    obs_mapping_keys: List[str] = ["leiden", "Cell type annotation"],
    fate_key: str = "Cell type annotation",
    use_key: str = "X_pca",
    return_adata: bool = True,
    normalize_potential: bool = True,
    potential_normalization_kwargs={},
    name: Optional[str] = None,
    save_h5ad: bool = False,
    gene_ids_key: str = "gene_ids",
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    dt: float = 0.1,
    N: int = 2000,
    device: Union[torch.device, str] = autodevice.AutoDevice(),
    time_key: str = "Time point",
    ref_cell_type_key="Cell type annotation",
    time_key_added: str = "t",
    simulation_key_added: str = "simulation",
    final_state_key_added: str = "final_state",
    silent: bool = False,
    graph=None,
    ref_kNN_key: str = "X_pca",
    PCA=None,
    UMAP=None,
    wd=".",
    return_simulator=False,
    gpu: bool = True,
    *args,
    **kwargs
) -> anndata.AnnData:

    """
    Parameters:
    -----------

    Returns:
    --------
    """

    KWARGS = utils.function_kwargs(Simulator, locals())
    KWARGS.update(kwargs)
    simulator = Simulator(**KWARGS)
    
    DiffEq = model.DiffEq
    KWARGS = utils.function_kwargs(simulator, locals())

    adata_sim = simulator(**KWARGS)

    if return_simulator:
        return adata_sim, simulator

    return adata_sim