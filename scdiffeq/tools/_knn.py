
from ..core import utils

from ._x_use import fetch_formatted_data

import anndata
import pandas as pd
import numpy as np
import annoy


from typing import Union, List, Dict
NoneType = type(None)


class NeighborCounter(utils.ABCParse):
    def __init__(self, adata, X_neighbors, max_only: bool = False):

        self.__parse__(locals(), public=[None])

    @property
    def _N_NEIGHBORS(self):
        return self._X_neighbors.shape[1]

    @property
    def _NN_ADATA(self):
        if not hasattr(self, "_nn_adata"):
            self._nn_adata = self._adata[self._X_neighbors.flatten()]
        return self._nn_adata

    def forward(self, obs_key):
        labelled_neighbor_df = pd.DataFrame(
            self._NN_ADATA.obs[obs_key].to_numpy().reshape(-1, self._N_NEIGHBORS).T
        )
        return self._count_values(labelled_neighbor_df)

    def _to_list(self, obs_keys):
        if isinstance(obs_keys, str):
            obs_keys = list(obs_keys)
        return obs_keys

    def _count_values_enumerate(self, col: pd.Series) -> dict:
        return col.value_counts().to_dict()

    def _count_values_max(self, col: pd.Series) -> str:
        return col.value_counts().idxmax()

    def _count_values(self, labelled_neighbor_df: pd.DataFrame) -> List[Dict]:
        """
        labelled_neighbor_df
        """
        if not self._max_only:
            func = self._count_values_enumerate
        else:
            func = self._count_values_max

        counts = [
            func(labelled_neighbor_df[col]) for col in labelled_neighbor_df.columns
        ]

        if not self._max_only:
            return pd.DataFrame(counts).fillna(0)
        return counts

    def __call__(
        self, obs_keys: Union[List[str], str]
    ) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:

        """"""

        obs_keys = self._to_list(obs_keys)

        Counts = {}
        for obs_key in obs_keys:
            Counts[obs_key] = self.forward(obs_key)

        if self._max_only:
            return pd.DataFrame(Counts)
        return Counts


class kNN(utils.ABCParse):
    _IDX_BUILT = False

    def __init__(
        self,
        adata: anndata.AnnData,
        use_key: str = "X_pca",
        metric: str = "euclidean",
        n_neighbors: int = 20,
        n_trees: int = 10,
        *args,
        **kwargs,
    ):
        self.__parse__(locals(), public=[None])

    @property
    def X_basis(self):
        """Basis data on which the graph is built."""
        if not hasattr(self, "_X_basis"):
            self._X_basis = fetch_formatted_data(
                self._adata, use_key=self._use_key, torch=False
            )

        return self._X_basis

    @property
    def _N_CELLS(self) -> int:
        return self.X_basis.shape[0]

    @property
    def _N_DIMS(self) -> int:
        return self.X_basis.shape[1]

    def _build_annoy_index(self) -> annoy.AnnoyIndex:

        idx = annoy.AnnoyIndex(self._N_DIMS, self._metric)
        [idx.add_item(i, self.X_basis[i]) for i in range(self._N_CELLS)]
        idx.build(self._n_trees)

        return idx

    @property
    def idx(self) -> annoy.AnnoyIndex:

        """"""

        if not hasattr(self, "_idx"):
            self._idx = self._build_annoy_index()
        return self._idx

    def query(self, X_query: np.ndarray) -> np.ndarray:
        """
        Key function to search for neighbor cells of given observations.

        Parameters:
        -----------
        X_query: np.ndarray
            Observations of size: [cells x n_dims].

        Returns:
        --------
        np.ndarray
            Query results composed neighbors.
        """

        return np.array(
            [
                self.idx.get_nns_by_vector(X_query[i], self._n_neighbors)
                for i in range(X_query.shape[0])
            ]
        )

    def count(
        self,
        X_neighbors: np.ndarray,
        obs_keys: Union[List[str], str],
        max_only: bool = False,
    ) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:

        """
        Count neighbor labels. If `max_only` is True. Provide the predicted label rather than value counts.

        Parameters
        ----------
        X_neighbors: np.ndarray

        obs_keys: Union[List[str], str]

        max_only: bool

        Returns
        -------

        """

        neighbor_counter = NeighborCounter(self._adata, X_neighbors, max_only=max_only)
        return neighbor_counter(obs_keys)

    def _aggregate(
        self, X_query, obs_keys: Union[List[str], str], max_only: bool = False
    ) -> pd.DataFrame:
        """Queries and counts"""

        X_nn = self.query(X_query)
        return self.count(X_nn, obs_keys=obs_keys, max_only=max_only)

    def _multi_aggregate(
        self, X_query, obs_keys: str, max_only: bool = False
    ) -> List[pd.DataFrame]:

        """
        Queries and counts across a stacked (3-D) X_query input.
        """
        _df_list = [
            self.aggregate(X_query=X_, obs_keys=obs_keys, max_only=max_only)
            for X_ in X_query
        ]
        return _df_list

    def aggregate(
        self, X_query, obs_keys: str, max_only: bool = False
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """"""

        KWARGS = {"X_query": X_query, "obs_keys": obs_keys, "max_only": max_only}

        if X_query.ndim == 3:
            return self._multi_aggregate(**KWARGS)
        if X_query.ndim == 2:
            return self._aggregate(**KWARGS)
        raise ValueError(f"X_query.ndim should be 2 or 3. Received: {X_query.ndim}")

    def __repr__(self):
        return f"kNN: {self._use_key}"
