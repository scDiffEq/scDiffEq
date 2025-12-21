
# -- import packages: ----------------------------------------------------------
import numpy as np
import anndata
import ABCParse


# -- import local dependencies: ------------------------------------------------
from ...core import utils


class MeshGrid(ABCParse.ABCParse):
    def __init__(self, adata: anndata.AnnData, gridpoints: int = 250, use_key: str = "X_umap"):

        self.__parse__(locals(), public=[None])

    @property
    def X_umap(self):
        return self._adata.obsm[self._use_key]

    @property
    def _X_MIN(self):
        return np.min(self.X_umap[:, 0])

    @property
    def _X_MAX(self):
        return np.max(self.X_umap[:, 0])

    @property
    def _Y_MIN(self):
        return np.min(self.X_umap[:, 1])

    @property
    def _Y_MAX(self):
        return np.max(self.X_umap[:, 0])

    @property
    def _X_GRID(self):
        return np.linspace(self._X_MIN, self._X_MAX, self._gridpoints)

    @property
    def _Y_GRID(self):
        return np.linspace(self._Y_MIN, self._Y_MAX, self._gridpoints)

    def __call__(self):
        return np.meshgrid(self._X_GRID, self._Y_GRID)