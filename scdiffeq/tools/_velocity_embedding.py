import warnings
import adata_query

import ABCParse
import numpy as np
import anndata


from .utils import L2Norm, transition_matrix

class VelocityEmbedding(ABCParse.ABCParse):
    """Compute X_emb and V_emb"""

    def __init__(
        self,
        velocity_key: str = "velocity",
        self_transitions: bool = True,
        use_negative_cosines: bool = True,
        T_scale: float = 10,
        *args,
        **kwargs,
    ):
        self.__parse__(locals())

        self._L2Norm = L2Norm()

    @property
    def X_emb(self):
        if not hasattr(self, "_X_emb"):
            self._X_emb = adata_query.fetch(
                self._adata, key=f"X_{self._basis}", torch=False
            )
        return self._X_emb

    @property
    def T(self):
        if not hasattr(self, "_T"):
            self._T = transition_matrix(
                adata=self._adata,
                self_transitions=self._self_transitions,
                use_negative_cosines=self._use_negative_cosines,
                scale=self._T_scale,
                force_dense=False,
                disable_dense=True,
                return_cls=False,
            )
        return self._T

    def forward(self, i):
        # get indices of the ith-cell transitions
        indices = self.T[i].indices
        # compute 2-D coordinate-difference (dX) b/t neighbors
        dX = self.X_emb[indices] - self.X_emb[i, None]  # shape (n_neighbors, 2)

        if not self._retain_scale:
            dX /= self._L2Norm(dX)[:, None]

        # In steady state, dX = 0
        dX[np.isnan(dX)] = 0
        prob = self.T[i].data
        self._V_emb[i] = prob.dot(dX) - prob.mean() * dX.sum(0)

    def __call__(
        self,
        adata: anndata.AnnData,
        basis: str = "umap",
        retain_scale: bool = False,
        *args,
        **kwargs,
    ):
        """Compute the velocity embedding (V_emb)

        Args:
            adata (anndata.AnnData)

        Returns:
            V_emb (np.ndarray): Velocity embedding (n_cells, 2)
        """
        self.__update__(locals())

        self._V_emb = np.zeros_like(self.X_emb)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i in range(self._adata.n_obs):
                self.forward(i)

        return self.X_emb, self._V_emb
