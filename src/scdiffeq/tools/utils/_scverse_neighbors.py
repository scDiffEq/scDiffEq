# -- import packages: ---------------------------------------------------------
import ABCParse
import adata_query
import anndata
import logging
import scanpy as sc

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -- set type hints: ----------------------------------------------------------
from typing import Any, Dict, Optional, Union


# -- Operational class: -------------------------------------------------------
class SCVerseNeighbors(ABCParse.ABCParse):
    def __init__(
        self,
        distances_key: str = "distances",
        connectivities_key: str = "connectivities",
        params_key: str = "neighbors",
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            distances_key (Optional[str]): Key accessor to cell neighbor
            distances in ``adata.obsp``. **Default**: "distances".
            connectivities_key (Optional[str]): Key accessor to cell neighbor
            connectivities in ``adata.obsp``. **Default**: "connectivities".
            params_key (str): **Default**: "neighbors".

        Returns:
            None
        """
        self.__parse__(locals())

        self._neighbor_computation_performed = False

    @property
    def _SCANPY_NEIGHBORS_KWARGS(self) -> Dict[str, Any]:
        return ABCParse.function_kwargs(sc.pp.neighbors, self._PARAMS)

    @property
    def _HAS_DISTANCES(self) -> bool:
        return self._distances_key in self._adata.obsp

    @property
    def _HAS_CONNECTIVITIES(self) -> bool:
        return self._connectivities_key in self._adata.obsp

    @property
    def _HAS_NN_PARAMS(self) -> bool:
        return self._params_key in self._adata.uns

    def _probable_fetch(self, key):
        return adata_query.fetch(self._adata, key)

    @property
    def distances(self):
        if self._HAS_DISTANCES:
            return self._probable_fetch(self._distances_key)

    @property
    def connectivities(self):
        if self._HAS_CONNECTIVITIES:
            return self._probable_fetch(self._connectivities_key)

    @property
    def params(self):
        if self._HAS_NN_PARAMS:
            return self._probable_fetch(self._params_key)

    @property
    def properties(self):
        return {
            attr[1:]: getattr(self, attr) for attr in self.__dir__() if "_HAS_" in attr
        }

    @property
    def neighbors_precomputed(self) -> bool:
        return all(self.properties.values())

    def _intake(self, adata: anndata.AnnData) -> None:
        """"""

        self._preexisting = {}

        for key in ["obsp", "uns"]:
            if not hasattr(adata, key):
                self._preexisting[key] = None
            else:
                self._preexisting[key] = list(getattr(adata, key).keys())

    def _message(self, adata: anndata.AnnData) -> None:
        for key, val in self._preexisting.items():
            added = [attr for attr in getattr(adata, key).keys() if not attr in val]
            for added_val in added:
                logger.info(f"Added: adata.{key}['{added_val}']")

    def forward(self, adata: anndata.AnnData, **kwargs) -> None:
        """"""
        self.__update__(kwargs)

        self._intake(adata)

        if self.neighbors_precomputed and self._force:
            logger.info("Force recomputing neighbors")

        if not self.neighbors_precomputed or self._force:
            kw = self._SCANPY_NEIGHBORS_KWARGS.copy()
            kw.update({"adata": adata})

            sc.pp.neighbors(**kw)

            self._neighbor_computation_performed = True
        self._message(adata)

    def __call__(
        self,
        adata: anndata.AnnData,
        n_neighbors: int = 15,
        n_pcs: Optional[int] = None,
        use_rep: Optional[str] = None,
        random_state: Optional[Union[int, None]] = 0,
        method: Optional[str] = 'umap',
        metric: Optional[str] = 'euclidean',
        distances_key: Optional[str] = "distances",
        connectivities_key: Optional[str] = "connectivities",
        params_key: Optional[str] = "neighbors",
        force: bool = False,
        silent: Optional[bool] = False,
        return_cls: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> None:
        self.__update__(locals())

        self.forward(adata)


# -- API-facing function: -----------------------------------------------------
def scverse_neighbors(
    adata: anndata.AnnData,
    n_neighbors: int = 15,
    n_pcs: Optional[int] = None,
    use_rep: Optional[str] = None,
    random_state: Optional[Union[int, None]] = 0,
    method: Optional = "umap",
    metric: Optional = "euclidean",
    distances_key: Optional[str] = "distances",
    connectivities_key: Optional[str] = "connectivities",
    params_key: Optional[str] = "neighbors",
    force: Optional[bool] = False,
    return_cls: Optional[bool] = False,
    *args,
    **kwargs,
):
    """Scanpy's ``sc.pp.neighbors`` with a few book-keeping functions added.

    Args:
        adata: anndata.AnnData

        n_neighbors (int): decsription. **Default** = 15

        distances_key (str): decsription. **Default** = "distances"

        connectivities_key (str): decsription. **Default** = "connectivities"

        params_key (str): decsription. **Default** = "neighbors"

        force (bool): decsription. **Default** = False

        return_cls (bool): Return the operator class for access to functional handles. **Default** = False

    Returns:
        None, updates ``adata``.
    """

    scv_neighbors = SCVerseNeighbors(
        distances_key=distances_key,
        connectivities_key=connectivities_key,
        params_key=params_key,
    )

    funcs = [sc.pp.neighbors, scv_neighbors.__call__]

    for func in funcs:
        kwargs.update(ABCParse.function_kwargs(func, locals()))

    scv_neighbors(**kwargs)

    if return_cls:
        return scv_neighbors
