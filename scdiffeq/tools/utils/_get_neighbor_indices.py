
# -- import packages: ---------------------------------------------------------
import ABCParse
import anndata
import numpy as np
import scipy.sparse


# -- Set typing: --------------------------------------------------------------
from typing import Optional


# -- Controller class: --------------------------------------------------------
class DistancesHandler(ABCParse.ABCParse):
    def __init__(self, distances_key: str = "distances", *args, **kwargs):
        """
        Args:
            distances_key (str)

        """
        self.__parse__(locals())

    @property
    def distances(self) -> scipy.sparse.csr_matrix:
        if not hasattr(self, "_distances"):
            D = self._adata.obsp[self._distances_key].copy()
            D.data += 1e-6
            self._distances = D
        return self._distances

    @property
    def n_counts(self):
        return (self.distances > 0).sum(axis=1).A.flatten()

    @property
    def n_neighbors(self):
        if not hasattr(self, "_n_neighbors") or self._n_neighbors is None:
            return self.n_counts.min()
        if self.n_counts.min() == 0:
            return self._n_neighbors
        return min(self.n_counts.min(), self._n_neighbors)

    @property
    def rows(self):
        return np.where(self.n_counts > self.n_neighbors)[0]

    @property
    def cumsum_neighbors(self):
        return np.insert(self.n_counts.cumsum(), 0, 0)

    @property
    def _data(self):
        if not hasattr(self, "_private_data"):
            self._private_data = self.distances.data
        return self._private_data

    def _idx_filter(self, row):
        """clean up indices of a row if criteria are met before
        passing to this function"""
        n0, n1 = self.cumsum_neighbors[row], self.cumsum_neighbors[row + 1]
        rm_idx = self._data[n0:n1].argsort()[self.n_neighbors :]
        self._data[rm_idx] = 0

    def _return_mode_distances(self):
        """scvelo also enables return mode on connectivities. we'll
        forego that for now."""
        return self.distances.indices.reshape((-1, self.n_neighbors))

    def __call__(
        self, adata: anndata.AnnData, n_neighbors: Optional[int] = None, *args, **kwargs
    ):
        """Return distance matrix indices

        Args:
            adata (anndata.AnnData)

            n_neighbors (Optional[int])
        """
        self.__update__(locals())

        _ = [self._idx_filter(row) for row in self.rows]
        self.distances.eliminate_zeros()
        self.distances.data -= 1e-6

        return self._return_mode_distances()


# -- API-facing function: -----------------------------------------------------
def get_neighbor_indices(
    adata: anndata.AnnData,
    n_neighbors: Optional[int] = None,
    distances_key: Optional[str] = "distances",
):
    """Return distance matrix indices

    Args:
        adata (anndata.AnnData): anndata.

        n_neighbors (Optional[int]): 
        
        distances_key (Optional[str]): Key accessor to cell neighbor distances
        in ``adata.obsp``. **Default**: "distances".

    Returns:
        (np.ndarray)

    """

    distances_handler = DistancesHandler(distances_key=distances_key)
    return distances_handler(adata=adata, n_neighbors=n_neighbors)
