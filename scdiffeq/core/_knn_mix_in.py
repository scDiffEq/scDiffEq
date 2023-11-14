
import anndata
import ABCParse
from typing import Optional

from .. import tools

class kNNMixIn(ABCParse.ABCParse, object):
    def __init__(self, *args, **kwargs):
        self.__parse__(locals())
        
    @property
    def _adata_kNN_fit(self):
        return self.adata[self.adata.obs[self._kNN_fit_subset]].copy()
        
    def _kNN_info_msg(self):
        
        self._INFO(f"Bulding Annoy kNN Graph on adata.obsm['{self._kNN_fit_subset}']")
        
    def configure_kNN(
        self,
        adata: Optional[anndata.AnnData] = None,
        kNN_key: Optional[str] = None,
        kNN_fit_subset: Optional[str] = None,
    ):
        
        """
        subset key should point to a col in adata.obs of bool vals
        """
        self.__update__(locals())
        
        self._kNN_info_msg()
        self._kNN = tools.kNN(
            adata = self._adata_kNN_fit, use_key = self._kNN_key,
        )

    @property
    def kNN(self):
        if not hasattr(self, "_kNN"):
            self.configure_kNN(
                adata = self._adata_kNN_fit,
                kNN_key = self._kNN_key,
                kNN_fit_subset = self._kNN_fit_subset,
            )
        return self._kNN