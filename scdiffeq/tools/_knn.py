
from ..core import utils

from ._x_use import X_use

import anndata as a
import pandas as pd
import numpy as np
import annoy


from typing import List
NoneType = type(None)


class kNN(utils.ABCParse):
    _IDX_BUILT = False

    def __init__(
        self,
        adata: a.AnnData,
        use_key: str = "X_pca",
        metric: str = "euclidean",
        n_neighbors: int = 20,
        n_trees: int = 10,
        *args,
        **kwargs,
    ):
        self.__parse__(locals(), public=[None])
        self.idx = self._build()

    @property
    def X_use(self):
        if not hasattr(self, "_X_use"):
            xu = X_use(self._adata, self._use_key)
            self._X_use = xu(torch = False, groupby = None, device = "cpu")
            
        return self._X_use

    @property
    def _N_CELLS(self):
        return self.X_use.shape[0]

    @property
    def _N_DIMS(self):
        return self.X_use.shape[1]

    def _build(self) -> annoy.AnnoyIndex:
        idx = annoy.AnnoyIndex(self._N_DIMS, self._metric)
        [idx.add_item(i, self.X_use[i]) for i in range(self._N_CELLS)]
        idx.build(self._n_trees)
        return idx

    def query(self, X_query: np.ndarray) -> np.ndarray:
        return np.array(
            [
                self.idx.get_nns_by_vector(X_query[i], self._n_neighbors)
                for i in range(X_query.shape[0])
            ]
        )

    def _count_values(self, col: pd.Series) -> dict:
        return col.value_counts().to_dict()

    def _max_count(self, col: pd.Series) -> str:
        return col.value_counts().idxmax()

    def count(self, query_result, obs_key: str, max_only: bool = False):
        nn_adata = self._adata[query_result.flatten()]

        query_df = pd.DataFrame(
            nn_adata.obs[obs_key].to_numpy().reshape(-1, self._n_neighbors).T
        )
        del nn_adata

        if not max_only:
            return [
                self._count_values(query_df[i]) for i in query_df.columns
            ]  # list of dicts
        return [
            self._max_count(query_df[i]) for i in query_df.columns
        ]  # list of values

    def aggregate(self, X_query, obs_key: str, max_only: bool = False):
        _df = (
            pd.DataFrame(
                self.count(
                    query_result=self.query(X_query=X_query),
                    obs_key=obs_key,
                    max_only=max_only,
                )
            )
            .fillna(0)
            .sort_index(axis=1)
        )
        if not max_only:
            return _df
        return _df.rename({0:obs_key}, axis = 1)
            

    def multi_aggregate(
        self, X_query, obs_key: str, max_only: bool = False
    ) -> List[pd.DataFrame]:
        _list_of_dfs = [
            self.aggregate(
                X_query=X_query[i],
                obs_key=obs_key,
                max_only=max_only,
            )
            for i in range(len(X_query))
        ]
        
        if max_only:
            concat_df = pd.concat(_list_of_dfs, axis = 1)
            concat_df.columns = range(len(X_query))
            return concat_df
        
        return _list_of_dfs

    def __repr__(self):
        return f"kNN: {self._use_key}"