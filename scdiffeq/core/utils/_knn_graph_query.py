# -- import packages: ----------------------------------------------------------
import ABCParse
import annoyance
import logging
import pandas as pd
import torch

# -- import local dependencies: -----------------------------------------------
from ._anndata_inspector import AnnDataInspector
from ._function_kwargs import extract_func_kwargs

# -- configure logging: --------------------------------------------------------
logger = logging.getLogger(__name__)


# -- operational cls: ----------------------------------------------------------
class kNNGraphQuery(ABCParse.ABCParse):
    def __init__(self, adata, use_key, *args, **kwargs):

        self.__parse__(locals())
        self._IGNORE += ["X_hat", "query_t"]

        self._configure_graph()

    def configure_used_layer(self):
        inspector = AnnDataInspector(self.adata)
        if self.use_key in inspector.layers:
            self.adata.obsm[self.use_key] = self.adata.layers[self.use_key]

    @property
    def _GRAPH_KWARGS(self):
        return extract_func_kwargs(annoyance.kNN, self._PARAMS)

    def _configure_graph(self):

        self.configure_used_layer()
        self.Graph = annoyance.kNN(**self._GRAPH_KWARGS)
        self.Graph.build()

    def _format_input_shape(self, X_hat, query_t):
        """"""
        if not query_t is None:
            return X_hat[query_t][None, :]
        return X_hat

    def _format_input_device(self, X_hat: torch.Tensor):

        if X_hat.device != "cpu":
            return X_hat
        return X_hat.detach().cpu().numpy()

    def _format_input(self, X_hat, query_t):
        return self._format_input_device(self._format_input_shape(X_hat, query_t))

    def _count(self, X_nn):
        return (
            self.adata[X_nn.flatten().astype(str)]
            .obs[self.annot_key]
            .values.reshape(-1, self.Graph._n_neighbors)
        )

    def _query(self, X_hat):
        return self._count(self.Graph.query(X_hat))

    def _fate_df(self, x_lab):
        return pd.DataFrame([x_lab[0].value_counts() for i in range(len(x_lab))])

    def __call__(self, X_hat: torch.Tensor, annot_key: str, query_t: int = -1):
        """
        X_hat: torch.Tensor of shape: [t, n_sim, n_dim]
        """
        self.__update__(locals())

        X_hat = self._format_input(X_hat, query_t)
        return pd.DataFrame([self._fate_df(self._query(X)).idxmax(1) for X in X_hat]).T
