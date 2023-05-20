
import numpy as np
import torch


class AnnDataInspector:
    def __init__(self, adata):

        self._adata = adata

    @property
    def layers(self):
        return list(self._adata.layers)

    @property
    def obsm(self):
        return self._adata.obsm_keys()

    @property
    def attributes(self):
        return ["X"] + self.layers + self.obsm

    @property
    def is_np_array(self):
        return isinstance(self._X_use, np.ndarray)

    def as_tensor(self):
        return torch.Tensor(self._X_use)

    def _isolate_data_matrix(self, key):
        if key in self.layers:
            return self._adata.layers[key]
        elif key in self.obsm:
            return self._adata.obsm[key]

    def __call__(self, key, as_tensor=True):

        self._X_use = self._isolate_data_matrix(key)
        if self.is_np_array:
            if as_tensor:
                return self.as_tensor()
            return self._X_use