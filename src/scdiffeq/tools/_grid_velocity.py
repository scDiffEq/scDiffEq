
# -- import packages: --------------------------------------------------
import ABCParse
import adata_query
import anndata
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sklearn.neighbors


# -- set typing: -------------------------------------------------------
from typing import Optional

import ABCParse
import numpy as np

# -----------------------------------------------------------------------------
# class Quiver(ABCParse.ABCParse):
#     """With autoscaling"""

#     def __init__(self, *args, **kwargs):
#         self.__parse__(locals())

#     @property
#     def _scale_factor(self):
#         """enables handling large values"""
#         return np.abs(self._X_emb).max()

#     def quiver(self, ax):
#         ax.quiver(
#             self._X_emb[:, 0] / self._scale_factor,
#             self._X_emb[:, 1] / self._scale_factor,
#             self._V_emb[:, 0],
#             self._V_emb[:, 1],
#             angles="xy",
#             scale_units="xy",
#             scale=None,
#         )

#     def __call__(self, ax, X_emb: np.ndarray, V_emb: np.ndarray):
#         """
#         X_emb (np.ndarray): state embedding.
#         V_emb (np.ndarray): velocity embedding.
#         """

#         self.__update__(locals())

#         self.quiver(ax)
   

# -- supporting function: --------------------------------------------------
def quiver_autoscale(X_emb: np.ndarray, V_emb: np.ndarray):
    """ """
    
    scale_factor = np.abs(X_emb).max()  # just so that it handles very large values
    fig, ax = plt.subplots()
    Q = ax.quiver(
        X_emb[:, 0] / scale_factor,
        X_emb[:, 1] / scale_factor,
        V_emb[:, 0],
        V_emb[:, 1],
        angles="xy",
        scale_units="xy",
        scale=None,
    )
    Q._init()
    fig.clf()
    plt.close(fig)
    return Q.scale / scale_factor


# -- operational class: -------------------------------------------------------
class GridVelocity(ABCParse.ABCParse):
    def __init__(
        self,
        density: float = 1,
        smooth: float = 0.5,
        n_neighbors: Optional[int] = None,
        min_mass: float = 1,
        autoscale=True,
        stream_adjust=True,
        cutoff_percentile: float = 0.05,
        *args,
        **kwargs
    ):
        
        self.__parse__(locals())

    @property
    def valid_idx(self):
        """use to rm invalid cells"""
        if not hasattr(self, "_valid_idx"):
            self._valid_idx = np.isfinite(self._X_emb.sum(1) + self._V_emb.sum(1))
        return self._valid_idx

    @property
    def X_emb(self):
        return self._X_emb[self.valid_idx]

    @property
    def V_emb(self):
        return self._V_emb[self.valid_idx]

    @property
    def n_obs(self):
        return self.X_emb.shape[0]

    @property
    def n_dim(self):
        return self.X_emb.shape[1]

    def _create_mesh_grid(self):
        """returns X_grid"""
        self._grs = []
        for dim_i in range(self.n_dim):
            m, M = np.min(self.X_emb[:, dim_i]), np.max(self.X_emb[:, dim_i])
            m = m - 0.01 * np.abs(M - m)
            M = M + 0.01 * np.abs(M - m)
            self._grs.append(np.linspace(m, M, int(50 * self._density)))

        return np.vstack([i.flat for i in np.meshgrid(*self._grs)]).T

    @property
    def n_neighbors(self):
        if not hasattr(self, "_n_neighbors") or self._n_neighbors is None:
            self._n_neighbors = int(self.n_obs / 50)
        return self._n_neighbors

    def _smooth_grid_neighbor_graph(self, X_emb, X_grid):
        """Fit ANN model to X_emb. Predict neighbors/dist for X_grid"""
        self._nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=self.n_neighbors, n_jobs=-1
        )
        self._nn.fit(X_emb)
        self._dists, self._neighbors = self._nn.kneighbors(X_grid)

    @property
    def scale(self):
        if not hasattr(self, "_scale"):
            self._scale = np.mean([(g[1] - g[0]) for g in self._grs]) * self._smooth
        return self._scale

    @property
    def weight(self):
        return scipy.stats.norm.pdf(x=self._dists, scale=self.scale)

    @property
    def p_mass(self):
        return self.weight.sum(1)

    def _estimate_grid_velocities(self, X_grid):
        self._smooth_grid_neighbor_graph(self.X_emb, X_grid)

        V_grid = (self.V_emb[self._neighbors] * self.weight[:, :, None]).sum(1)
        V_grid /= np.maximum(1, self.p_mass)[:, None]

        return V_grid

    def do_stream_adjust_X_grid(self, X_grid):
        return np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])

    def do_stream_adjust_V_grid(self, V_grid: np.ndarray) -> np.ndarray:
        """"""
        ns = int(np.sqrt(len(V_grid[:, 0])))
        V_grid = V_grid.T.reshape(2, ns, ns)
        mass = np.sqrt((V_grid**2).sum(0))
        min_mass = 10 ** (self._min_mass - 6)  # default min_mass = 1e-5
        min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)

        cutoff = mass.reshape(V_grid[0].shape) < min_mass
        length = np.sum(np.mean(np.abs(self.V_emb[self._neighbors]), axis=1), axis=1).T
        length = length.reshape(ns, ns)
        cutoff |= length < np.percentile(length, self._cutoff_percentile)

        V_grid[0][cutoff] = np.nan

        return V_grid

    def __call__(self, X_emb: np.ndarray, V_emb: np.ndarray, *args, **kwargs):
        self.__update__(locals())

        X_grid = self._create_mesh_grid()
        V_grid = self._estimate_grid_velocities(X_grid)

        if self._stream_adjust:
            X_grid = self.do_stream_adjust_X_grid(X_grid)
            V_grid = self.do_stream_adjust_V_grid(V_grid)
        else:
            # filter on mass
            min_mass *= np.percentile(self.p_mass, 99) / 100
            X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]

            if self._autoscale:
                V_grid /= 3 * quiver_autoscale(X_grid, V_grid)

        return X_grid, V_grid
