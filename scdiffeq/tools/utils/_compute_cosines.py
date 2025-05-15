# -- import packages: ---------------------------------------------------------
import ABCParse
import adata_query
import anndata
import logging
import numpy as np
import scipy.sparse
import warnings
import time

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
        init_start = time.time()
        self.__parse__(locals())

        self._L2Norm = L2Norm()
        self._cosine_correlation = CosineCorrelation()
        self._logger = logging.getLogger(__name__)
        self._logger.debug("ComputeCosines: Initialization complete in %.2f seconds." % (time.time() - init_start))

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
            nn_start = time.time()
            self._nn_idx = get_neighbor_indices(
                adata=self._adata,
                n_neighbors=self.n_neighbors,
                distances_key=self._distances_key,
            )
            self._logger.debug("ComputeCosines: Neighbor indices calculated in %.2f seconds." % (time.time() - nn_start))
        return self._nn_idx

    def _contains_non_zero(self, obs_id: int) -> bool:
        return np.any(np.array([self.V[obs_id].min(), self.V[obs_id].max()]) != 0)

    def forward(self, obs_id) -> None:
        if self._contains_non_zero(obs_id):
            try:
                iter_nn_idx = get_iterative_indices(self.nn_idx, obs_id, 2, None)
                dX = self.X[iter_nn_idx] - self.X[obs_id, None]
                corr = self._cosine_correlation(dX, self.V[obs_id])
                self._VALS.extend(corr)
                self._ROWS.extend(np.ones(len(iter_nn_idx)) * obs_id)
                self._COLS.extend(iter_nn_idx)
            except Exception as e:
                # Log error but continue processing
                self._logger.warning(f"Error computing velocity graph for cell {obs_id}: {str(e)}")

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
        if not self._VALS:
            # Handle case with no valid values
            logger.warning("No valid correlations were computed. Creating empty graphs.")
            n = len(self.obs_idx)
            graph = scipy.sparse.csr_matrix((n, n))
            if self._split_negative:
                return graph, graph
            return graph
            
        self._VALS = self._replace_nans(self._VALS)

        graph = scipy.sparse.coo_matrix(
            (self._VALS, (self._ROWS, self._COLS)),
            shape=(len(self.obs_idx), len(self.obs_idx)),
        )
        if self._split_negative:
            return self._perform_split_negative(graph)
        return graph

    def _update_adata(
        self, graph: scipy.sparse.csr_matrix, graph_neg: scipy.sparse.csr_matrix = None
    ) -> None:

        pos_key = f"{self._velocity_key_added}_graph"
        
        if graph_neg is not None:
            neg_key = f"{self._velocity_key_added}_graph_neg"
            for key, obj in zip([pos_key, neg_key], [graph, graph_neg]):
                if key in self._adata.obsp:
                    self._adata.obsp[key] = obj
                    logger.info(f"Updated: adata.obsp['{key}']")
                else:
                    self._adata.obsp[key] = obj
                    logger.info(f"Added: adata.obsp['{key}']")
        else:
            if pos_key in self._adata.obsp:
                self._adata.obsp[pos_key] = graph
                logger.info(f"Updated: adata.obsp['{pos_key}']")
            else:
                self._adata.obsp[pos_key] = graph
                logger.info(f"Added: adata.obsp['{pos_key}']")

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

        loop_start = time.time()
        n_obs = len(self.obs_idx)
        for i, obs_id in enumerate(self.obs_idx):
            if i < 10 or (i+1) % 1000 == 0 or (i+1) == n_obs:
                step_start = time.time()
                gi_start = time.time()
                try:
                    iter_nn_idx = get_iterative_indices(self.nn_idx, obs_id, 2, None)
                except Exception as e:
                    self._logger.warning(f"Error in get_iterative_indices for cell {obs_id}: {str(e)}")
                    continue
                gi_time = time.time() - gi_start

                dx_start = time.time()
                dX = self.X[iter_nn_idx] - self.X[obs_id, None]
                dx_time = time.time() - dx_start

                cc_start = time.time()
                corr = self._cosine_correlation(dX, self.V[obs_id])
                cc_time = time.time() - cc_start

                self._VALS.extend(corr)
                self._ROWS.extend(np.ones(len(iter_nn_idx)) * obs_id)
                self._COLS.extend(iter_nn_idx)

                total_time = time.time() - step_start
                self._logger.debug(f"Cell {i+1}/{n_obs}: get_iterative_indices={gi_time:.4f}s, dX={dx_time:.4f}s, cosine_corr={cc_time:.4f}s, total={total_time:.4f}s")
            else:
                self.forward(obs_id)
            if (i+1) % 1000 == 0 or (i+1) == n_obs:
                self._logger.debug(f"ComputeCosines: Processed {i+1}/{n_obs} cells in {time.time() - loop_start:.2f} seconds.")

        self._logger.debug("ComputeCosines: Main loop finished in %.2f seconds." % (time.time() - loop_start))

        graph_assembly_start = time.time()
        graph_result = self._assemble_graph_as_csr()
        self._logger.debug("ComputeCosines: Graph assembly finished in %.2f seconds." % (time.time() - graph_assembly_start))
        self._update_adata(*graph_result)
