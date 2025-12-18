# -- import packages: ----------------------------------------------------------
import anndata
import annoyance
import logging
import numpy as np
import pandas as pd
import sklearn.decomposition
import torch

# -- import local dependencies: ------------------------------------------------
from ._anndata_inspector import AnnDataInspector

# -- set type hints: -----------------------------------------------------------
from typing import Optional

# -- configure logging: --------------------------------------------------------
logger = logging.getLogger(__name__)


# -- operational cls: ----------------------------------------------------------
class FastGraph:
    """ """

    def __init__(
        self,
        adata: anndata.AnnData,
        use_key: str,
        annot_key: str = "Cell type annotation",
    ) -> None:
        """ """

        self.adata = adata
        self._inspector = AnnDataInspector(adata)
        self.annot_key = annot_key
        self.use_key = use_key
        if use_key in self._inspector.layers:
            adata.obsm[use_key] = adata.layers[use_key]

        self.Graph = annoyance.kNN(adata, use_key=self.use_key)
        self.Graph.build()

    def _fast_count(self, X_nn):
        return (
            self.adata[X_nn.flatten().astype(str)]
            .obs[self.annot_key]
            .values.reshape(-1, self.Graph._n_neighbors)
        )

    def _query(self, X_fin):
        return self._fast_count(self.Graph.query(X_fin))

    def _fate_df(self, x_lab) -> pd.DataFrame:
        return pd.DataFrame([x_lab[0].value_counts() for i in range(len(x_lab))])

    def _DETACH(self, X_hat: torch.Tensor) -> np.ndarray:
        return X_hat.detach().cpu().numpy()

    def _DETACH_FINAL(self, X_hat: torch.Tensor) -> np.ndarray:
        return X_hat[-1].detach().cpu().numpy()

    def _TRANSFORM(self, dimension_reduction_model, X_fin):
        return dimension_reduction_model.transform(X_fin)

    def __call__(
        self,
        X_hat: torch.Tensor,
        dimension_reduction_model: Optional[sklearn.decomposition.PCA] = None,
        final_timepoint_only: bool = True,
    ) -> pd.DataFrame:

        if X_hat.device != "cpu" and (final_timepoint_only):
            X_hat_ = self._DETACH_FINAL(X_hat)
        elif X_hat.device != "cpu":
            X_hat_ = self._DETACH(X_hat)
        elif final_timepoint_only:
            X_hat_ = X_hat[-1].numpy()
        else:
            X_hat_ = X_hat.numpy()

        if not dimension_reduction_model is None:
            X_hat_ = self._TRANSFORM(dimension_reduction_model, X_fin=X_hat_)

        return self._fate_df(self._query(X_fin=X_hat_))
