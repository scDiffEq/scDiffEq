

from ...core import utils

import pandas as pd
import numpy as np
import ABCParse


NoneType = type(None)

class Scatter3D(ABCParse.ABCParse):
    def __init__(self, z_key="psi", use_key="X_umap", z_adj=0.01, color=None):

        self.__parse__(locals(), public=[None])

    @property
    def X_umap(self):
        return self.adata.obsm[self._use_key]

    @property
    def Z_smooth(self) -> np.ndarray:
        return self.neighbor_model.predict(self.X_umap)

    def _compose_scatter_dict(self):

        SCATTER_DICT = {"UMAP-1": self.X_umap[:, 0], "UMAP-2": self.X_umap[:, 1]}
        SCATTER_DICT[self._z_key] = self.Z_smooth + self._z_adj

        if not isinstance(self._color, NoneType):
            SCATTER_DICT[self._color] = self.adata_scatter.obs[self._color]

        return SCATTER_DICT

    @property
    def scatter_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._compose_scatter_dict())

    def __call__(self, adata, neighbor_model):
        self.__update__(locals())
        return self.scatter_frame