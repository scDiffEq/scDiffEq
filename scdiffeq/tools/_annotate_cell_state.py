# -- import packages: ---------------------------------------------------------
import ABCParse
import anndata
import adata_query
import logging
import numpy as np

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -- controlling class: -------------------------------------------------------
class CellStateAnnotation(ABCParse.ABCParse):
    """Annotate cell states using a kNN graph."""

    def __init__(self, kNN: "kNN", silent: bool = False, *args, **kwargs) -> None:
        """Initialize `CellStateAnnotation` class.

        Args:
            kNN ('kNN')
                k-nearest neighbor graph.

        Returns:
            None
        """
        self.__parse__(locals())

    @property
    def _REF_DIM(self) -> int:
        """basis dim of kNN graph"""
        return self._kNN.X_use.shape[1]

    @property
    def _QUERY_DIM(self) -> int:
        """n_dim of cell query"""
        return self.X_query.shape[1]

    @property
    def X_query(self) -> np.ndarray:
        """Pull the simulated query cell data."""
        if not hasattr(self, "_X"):
            self._X = adata_query.fetch(
                adata=self._adata_sim, key=self._use_key, torch=False
            )
        return self._X

    def _assert_dimension_check(self) -> None:
        """Ensure the passed query dimension matches the basis dimension of the kNN graph."""

        msg = f"Query dim from {self._use_key}: {self._QUERY_DIM} does not match the kNN graph basis dimension: {self._REF_DIM}"

        assert self._QUERY_DIM == self._REF_DIM, msg

    @property
    def X_mapped(self) -> np.ndarray:
        """Map simulated query cells to observed cell state neighors."""
        if not hasattr(self, "_X_mapped"):
            self._X_mapped = self._kNN.aggregate(
                X_query=self.X_query, obs_key=self._obs_key, max_only=True
            )
        return self._X_mapped

    def forward(self) -> None:
        """Add mapped cell state values to adata_sim.obs"""
        self._adata_sim.obs[self._obs_key] = self.X_mapped.values.flatten()
        if not self._silent:
            logger.info(f"Added state annotation: adata_sim.obs['{self._obs_key}']")

    def __call__(
        self,
        adata_sim: anndata.AnnData,
        obs_key: str,
        use_key: str = "X",
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            adata_sim (anndata.AnnData)

            obs_key (str)

            use_key (str)

        Returns:
            None
        """

        self.__update__(locals())

        self._assert_dimension_check()
        self.forward()


# -- API-facing function: -----------------------------------------------------
def annotate_cell_state(
    adata_sim: anndata.AnnData,
    kNN: "kNN",
    obs_key: str = "state",
    use_key: str = "X",
    silent: bool = False,
) -> None:
    """
    Use a kNN Graph to annotate simulated cell states.

    Parameters
    ----------
    adata_sim : anndata.AnnData
        Simulated data object in the format of ``anndata.AnnData``, the (annotated)
        single-cell data matrix of shape ``n_obs × n_vars``. Rows correspond to cells
        and columns to genes. For more: [1](https://anndata.readthedocs.io/en/latest/).

    kNN : kNN
        k-nearest neighbor graph.

    obs_key : str, optional
        Observation key. Default is "state".

    use_key : str, optional
        Key to use for the basis. Default is "X".

    silent : bool, optional
        If True, suppresses informational messages. Default is False.

    Returns
    -------
    None

    References
    ----------
    .. [1] https://anndata.readthedocs.io/en/latest/
    """

    state_annot = CellStateAnnotation(kNN=kNN, silent=silent)
    state_annot(adata_sim=adata_sim, obs_key=obs_key, use_key=use_key)
