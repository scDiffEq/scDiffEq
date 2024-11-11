import pandas as pd

# import larry
import numpy as np
import adata_query
import autodevice
import torch
import matplotlib.pyplot as plt
import tqdm.notebook
import ABCParse
import anndata
import lightning


from typing import Optional


class Simulation(ABCParse.ABCParse):
    """Sampled trajectories from an scDiffEq model"""

    def __init__(
        self,
        use_key: str = "X_pca",
        time_key: str = "Time point",
        N: int = 1,
        device: Optional[torch.device] = autodevice.AutoDevice(),
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

        self._T_GIVEN = False

    @property
    def _adata_init(self): ...

    @property
    def idx(self):
        """ """
        if not hasattr(self, "_idx"):
            self._idx = self._adata.obs.index
        return self._idx

    @property
    def Z0(self) -> torch.Tensor:
        """ """
        if not hasattr(self, "_Z0"):
            self._Z0 = adata_query.fetch(
                self._adata[self.idx],
                key=self._use_key,
                torch=True,
                device=self._device,
            )
            if self._N > 1:
                self._Z0 = self._Z0[None, :, :]
                self._Z0 = self._Z0.expand(self._N, -1, -1).flatten(0, 1)
        return self._Z0

    @property
    def _TIME(self) -> pd.Series:
        if not self._T_GIVEN:
            return self._adata.obs[self._time_key]
        return self.t

    @property
    def _T_MIN(self) -> float:
        """ """
        return self._TIME.min()

    @property
    def _T_MAX(self) -> float:
        """ """
        return self._TIME.max()

    @property
    def _N_STEPS(self) -> float:
        return int(((self._T_MAX - self._T_MIN) / self._dt) + 1)

    @property
    def t(self) -> torch.Tensor:
        """ """
        if not hasattr(self, "_t"):
            self._t = torch.linspace(
                self._T_MIN,
                self._T_MAX,
                self._N_STEPS,
            ).to(self._device)
        return self._t

    @property
    def _N_CELLS(self):
        return self._N * len(self.idx)

    def forward(self, Z0, t) -> torch.Tensor:
        return self._diffeq.forward(Z0, t).detach().cpu().flatten(0, 1).numpy()

    def _to_adata_sim(self, Z_hat: np.ndarray) -> anndata.AnnData:
        """
        Args:
            Z_hat (np.ndarray)

        Returns:
            adata_sim (anndata.AnnData)
        """
        adata_sim = anndata.AnnData(Z_hat)
        adata_sim.obs["t"] = np.repeat(self.t.detach().cpu().numpy(), self._N_CELLS)
        adata_sim.obs["z0_idx"] = np.tile(np.tile(self.idx, self._N_STEPS), self._N)
        adata_sim.obs["sim_i"] = np.tile(
            np.arange(self._N).repeat(len(self.idx)), self._N_STEPS
        )
        adata_sim.obs["sim"] = adata_sim.obs["z0_idx"].astype(str) + adata_sim.obs[
            "sim_i"
        ].astype(str)
        adata_sim.uns["sim_idx"] = self.idx
        adata_sim.uns["simulated"] = True

        return adata_sim

    def __call__(
        self,
        diffeq,
        adata: anndata.AnnData,
        idx: pd.Index,
        dt: float = 0.1,
        t: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> anndata.AnnData:
        """Simulate trajectories by sampling from an scDiffEq model.

        Args:
            diffeq (): lightning model.

            adata (anndata.AnnData): Input AnnDat object.

            idx (pd.Index): cell indices (corresponding to `adata` from which the model should
                initiate sampled trajectories.

        Returns:
            adata_sim (anndata.AnnData)
        """

        self.__update__(locals())

        self._diffeq.to(self._device)

        if not t is None:
            self._T_GIVEN = True

        Z_hat = self.forward(self.Z0, self.t)
        return self._to_adata_sim(Z_hat)


def simulate(
    adata: anndata.AnnData,
    diffeq: lightning.LightningModule,
    idx: Optional[pd.Index] = None,
    use_key: str = "X_pca",
    time_key: str = "Time point",
    N: Optional[int] = 1,
    t: Optional[torch.Tensor] = None,
    dt: Optional[float] = 0.1,
    device: Optional[torch.device] = autodevice.AutoDevice(),
    *args,
    **kwargs,
) -> anndata.AnnData:
    """
    Simulate trajectories by sampling from an scDiffEq model.

    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData object.
    idx : pd.Index
        Cell indices (corresponding to `adata`) from which the model should initiate sampled trajectories.
    diffeq : lightning.LightningModule
        The differential equation model.
    use_key : str, optional
        adata accession key for the input data. Default is "X_pca".
    N : int, optional
        Number of trajectories to sample from the model. Default is 2000.
    device : torch.device, optional
        Device to run the simulation on. Default is True.

    Returns
    -------
    anndata.AnnData
        AnnData object encapsulating scDiffEq model simulation.
    """
    if diffeq.__repr__() == "scDiffEq":
        diffeq = diffeq.DiffEq

    simulation = Simulation(
        use_key=use_key,
        time_key=time_key,
        N=N,
        device=device,
    )
    return simulation(diffeq=diffeq, adata=adata, idx=idx, t=t, dt=dt)
