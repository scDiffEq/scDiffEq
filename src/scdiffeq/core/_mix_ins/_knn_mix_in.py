# -- import packages: --------------------------------------------------------
import ABCParse
import anndata
import logging

# -- import local dependencies: -----------------------------------------------
from ... import tools

# -- set type hints: ----------------------------------------------------------
from typing import Optional


# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)


# -- mix-in cls: --------------------------------------------------------------
class kNNMixIn(ABCParse.ABCParse, object):
    def __init__(self, *args, **kwargs) -> None:
        self.__parse__(locals())

    @property
    def _adata_kNN_fit(self):
        return self.adata[self.adata.obs[self._kNN_fit_subset]].copy()

    def _kNN_info_msg(self) -> None:

        logger.info(f"Bulding Annoy kNN Graph on adata.obsm['{self._kNN_fit_subset}']")

    def configure_kNN(
        self,
        adata: Optional[anndata.AnnData] = None,
        kNN_key: Optional[str] = None,
        kNN_fit_subset: Optional[str] = None,
    ) -> None:
        """
        subset key should point to a col in adata.obs of bool vals
        """
        self.__update__(locals())

        logger.info(f"Bulding Annoy kNN Graph on adata.obsm['{self._kNN_fit_subset}']")

        self._kNN = tools.kNN(
            adata=self._adata_kNN_fit,
            use_key=self._kNN_key,
        )

    @property
    def kNN(self) -> tools.kNN:
        if not hasattr(self, "_kNN"):
            self.configure_kNN(
                adata=self._adata_kNN_fit,
                kNN_key=self._kNN_key,
                kNN_fit_subset=self._kNN_fit_subset,
            )
        return self._kNN
