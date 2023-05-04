
from ._info_message import InfoMessage
from ._anndata_inspector import AnnDataInspector

import annoyance
import pandas as pd

class FastGraph:
    def __init__(self, adata, use_key, annot_key="Cell type annotation"):

        self._INFO = InfoMessage()

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
        return self._fast_count(self.Graph.query(X_fin.detach().cpu().numpy()))

    def _fate_df(self, x_lab):
        return pd.DataFrame([x_lab[0].value_counts() for i in range(len(x_lab))])

    def __call__(self, X_hat):
        x_lab = self._query(X_hat[-1])
        return self._fate_df(x_lab)
