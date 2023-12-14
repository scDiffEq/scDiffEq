import pandas as pd
import larry
import numpy as np
import adata_query
import autodevice
import torch
import matplotlib.pyplot as plt
import tqdm.notebook
import ABCParse
import anndata


class Simulation(ABCParse.ABCParse):
    """Sampled trajectories from an scDiffEq model"""
    def __init__(
        self,
        use_key: str = "X_pca",
        N: int = 2000,
        device = autodevice.AutoDevice(),
        *args,
        **kwargs,
    ):
        """
        Args:
            use_key (str): description. **Default**: "X_pca"
            
            N (int): number of trajectories to sample from the model. **Default**: 2000
            
            device (bool): description. **Default**: "cuda:0"
            
        Returns:
            None
        """
        
        self.__parse__(locals())

    @property
    def _adata_init(self):
        ...

    @property
    def Z0(self) -> torch.Tensor:
        if not hasattr(self, "_Z0"):
            self._Z0 = adata_query.fetch(
                self._adata[self._idx],
                key=self._use_key,
                torch=True,
                device=self._device,
            ).expand(self._N, -1)
        return self._Z0

    @property
    def _T_MIN(self) -> float:
        return self._model.t_config.t_min

    @property
    def _T_MAX(self) -> float:
        return self._model.t_config.t_max

    @property
    def _N_STEPS(self) -> float:
        return self._model.t_config.n_steps

    @property
    def t(self) -> torch.Tensor:
        if not hasattr(self, "_t"):
            self._t = torch.linspace(
                self._T_MIN,
                self._T_MAX,
                self._N_STEPS,
            ).to(self._device)
        return self._t

    def forward(self, Z0, t) -> torch.Tensor:
        return self._model.DiffEq.forward(Z0, t).detach().cpu().flatten(0, 1).numpy()

    def _to_adata_sim(self, Z_hat) -> None:
        
        adata_sim = anndata.AnnData(Z_hat)
        adata_sim.obs["t"] = np.repeat(self.t.detach().cpu().numpy(), self._N)
        adata_sim.obs["sim"] = np.tile(range(self._N), self._N_STEPS)
        adata_sim.uns["sim_idx"] = self._idx
        adata_sim.uns["simulated"] = True

        return adata_sim

    def __call__(
        self, model: "scdiffeq.scDiffEq", adata: anndata.AnnData, idx: pd.Index, *args, **kwargs,
    ) -> anndata.AnnData:
        
        """Simulate trajectories by sampling from an scDiffEq model.

        Args:
            model (scdiffeq.scDiffEq): scDiffEq model. **Default**: ``True``.
            
            adata (anndata.AnnData): Input AnnDat object.

            idx (pd.Index): cell indices (corresponding to `adata` from which the model should
                initiate sampled trajectories.

        Returns:
            adata_sim (anndata.AnnData)
        """
        
        self.__update__(locals())
        
        self._model.to(self._device)

        Z_hat = self.forward(self.Z0, self.t)
        return self._to_adata_sim(Z_hat)


def simulate(
    adata: anndata.AnnData,
    idx: pd.Index,
    model: "scdiffeq.scDiffEq",
    use_key: str = "X_pca",
    device: torch.device = autodevice.AutoDevice(),
    N: int = 2000,
    *args,
    **kwargs
) -> anndata.AnnData:
    """Simulate trajectories by sampling from an scDiffEq model.

    Args:
        adata (anndata.AnnData): Input AnnDat object.

        idx (pd.Index): cell indices (corresponding to `adata` from which the model should
            initiate sampled trajectories.

        model (scdiffeq.scDiffEq): scDiffEq model. **Default**: ``True``.

        use_key (str): adata accession key for the input data. **Default**: "X_pca".

        N (int): Number of trajectories to sample from the model. **Default**: 2000.
        
        device (torch.device). description. **Default**: True.

    Returns:
        adata_sim (anndata.AnnData): AnnData object encapsulating scDiffEq model simulation.
    """
    
    simulation = Simulation(
        use_key=use_key,
        N=N,
        device=device,
    )
    return simulation(model=model, adata=adata, idx=idx)
