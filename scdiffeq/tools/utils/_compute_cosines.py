# -- import packages: ---------------------------------------------------------
import ABCParse
import adata_query
import anndata
import logging
import numpy as np
import scipy.sparse
import warnings

# -- import local dependencies: -----------------------------------------------
from ._norm import L2Norm
from ._get_neighbor_indices import get_neighbor_indices
from ._get_iterative_indices import get_iterative_indices

# -- set type hints: ----------------------------------------------------------
from typing import Optional

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -- supporting operational class: --------------------------------------------
class CosineCorrelation(ABCParse.ABCParse):
    def __init__(self, *args, **kwargs) -> None:
        """ """
        self.__parse__(locals())

        self._L2Norm = L2Norm()

    def _mean_subtraction(self, dX: np.ndarray) -> np.ndarray:
        return dX - dX.mean(-1)[:, None]

    def _Vi_norm(self, Vi: np.ndarray) -> np.ndarray:
        return self._L2Norm(Vi, axis=0)

    def forward(self, dX: np.ndarray, Vi: np.ndarray) -> np.ndarray:
        dx = self._mean_subtraction(dX)
        Vi_norm = self._Vi_norm(Vi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if Vi_norm == 0:
                return np.zeros(dx.shape[0])
            return (
                np.einsum("ij, j", dx, Vi)
                / (self._L2Norm(dx, axis=1) * Vi_norm)[None, :]
            )

    def __call__(self, dX: np.ndarray, Vi: np.ndarray, *args, **kwargs) -> np.ndarray:
        """ """
        self.__update__(locals())

        return self.forward(dX, Vi)


# -- operational class: -----------------------------------------------------------------
class ComputeCosines(ABCParse.ABCParse):
    def __init__(
        self,
        state_key="X",
        velocity_key: str = "X_drift",
        n_pcs: Optional[int] = None,
        split_negative: bool = True,
        distances_key: str = "distances", # NEW
        *args,
        **kwargs,
    ) -> None:
        self.__parse__(locals())

        self._L2Norm = L2Norm()
        self._cosine_correlation = CosineCorrelation()

    def _initialize_results(self) -> None:
        self._VALS, self._ROWS, self._COLS = [], [], []

    def _l2_norm(self, input):
        return self._L2Norm(input)

    @property
    def X(self):
        if not hasattr(self, "_X"):
            self._x = adata_query.fetch(self._adata, key=self._state_key, torch=False)
            if not self._n_pcs is None:
                self._x = self._x[:, : self._n_pcs]
        return self._x

    @property
    def V(self):
        if not hasattr(self, "_V"):
            self._v = adata_query.fetch(
                self._adata, key=self._velocity_key, torch=False
            )
            if not self._n_pcs is None:
                self._v = self._v[:, : self._n_pcs]
        return self._v

    @property
    def obs_idx(self):
        return range(len(self._adata))

    @property
    def n_neighbors(self) -> int:  # NEW
        if not hasattr(self, "_n_neighbors"):
            n_neighbor_param = self._adata.uns["neighbors"]["params"]["n_neighbors"]
            if isinstance(n_neighbor_param, int):
                self._n_neighbors = n_neighbor_param
            else:
                self._n_neighbors = n_neighbor_param[0]
        return self._n_neighbors

    @property
    def nn_idx(self):
        if not hasattr(self, "_nn_idx"):
            self._nn_idx = get_neighbor_indices(
                adata=self._adata,
                n_neighbors=self.n_neighbors,
                distances_key=self._distances_key,
            )
        return self._nn_idx

    def _contains_non_zero(self, obs_id: int) -> bool:
        return np.any(np.array([self.V[obs_id].min(), self.V[obs_id].max()]) != 0)

    def forward(self, obs_id) -> None:
        if self._contains_non_zero(obs_id):
            iter_nn_idx = get_iterative_indices(self.nn_idx, obs_id, 2, None)
            dX = self.X[iter_nn_idx] - self.X[obs_id, None]
            corr = self._cosine_correlation(dX, self.V[obs_id])
            self._VALS.extend(corr)
            self._ROWS.extend(np.ones(len(iter_nn_idx)) * obs_id)
            self._COLS.extend(iter_nn_idx)

    def _replace_nans(self, vals):
        vals = np.hstack(vals)
        vals[np.isnan(vals)] = 0
        return vals

    def _perform_split_negative(self, graph):
        graph_neg = graph.copy()

        graph.data = np.clip(graph.data, 0, 1)
        graph_neg.data = np.clip(graph_neg.data, -1, 0)

        graph.eliminate_zeros()
        graph_neg.eliminate_zeros()

        return graph.tocsr(), graph_neg.tocsr()

    def _assemble_graph_as_csr(self):
        self._VALS = self._replace_nans(self._VALS)

        graph = scipy.sparse.coo_matrix(
            (self._VALS, (self._ROWS, self._COLS)),
            shape=(len(self.obs_idx), len(self.obs_idx)),
        )
        if self._split_negative:
            return self._perform_split_negative(graph)
        return graph

    def _update_adata(
        self, graph: scipy.sparse.csr_matrix, graph_neg: scipy.sparse.csr_matrix
    ) -> None:

        pos_key = f"{self._velocity_key_added}_graph"
        neg_key = f"{self._velocity_key_added}_graph_neg"

        for key, obj in zip([pos_key, neg_key], [graph, graph_neg]):
            if key in self._adata.obsp:
                self._adata.obsp[key] = obj
                logger.info(f"Updated: adata.obsp['{key}']")
            else:
                self._adata.obsp[key] = obj
                logger.info(f"Added: adata.obsp['{key}']")

    def __call__(
        self,
        adata: anndata.AnnData,
        velocity_key_added: str = "velocity",
        *args,
        **kargs,
    ) -> None:
        """ """
        self.__update__(locals())

        self._initialize_results()

        for obs_id in self.obs_idx:
            self.forward(obs_id)

        self._update_adata(*self._assemble_graph_as_csr())
