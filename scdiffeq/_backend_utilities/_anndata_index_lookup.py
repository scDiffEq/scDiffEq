import abc
import numpy as np


class AnnDataIndexLookup(abc.ABC):
    def __init__(self): ...

    def _configure(self, adata):
        if not hasattr(self, "adata"):
            self.adata = adata.copy()

    @property
    @abc.abstractmethod
    def NAMES(self): ...

    def forward(self, idx):
        """Locate an individual cell index"""
        return np.where(idx == self.NAMES)[0]

    def __call__(self, adata, idx=[]) -> np.ndarray:
        self._configure(adata)
        return np.array([self.forward(ix) for ix in idx]).flatten()


class VarIndexLookUp(AnnDataIndexLookup):
    """
    At __call__, var_names are set and looked up.
    """

    def __init__(self, var_name_key: str = "gene_ids"):
        super().__init__()

        self._var_name_key = var_name_key

    def _set_var_names(self):
        self.adata.var_names = self.adata.var[self._var_name_key]

    def _configure(self, adata):
        if not hasattr(self, "adata"):
            self.adata = adata.copy()
            self._set_var_names()

    @property
    def NAMES(self):
        return self.adata.var_names


class ObsIndexLookUp(AnnDataIndexLookup):
    def __init__(self): ...

    @property
    def NAMES(self):
        return self.adata.obs_names
