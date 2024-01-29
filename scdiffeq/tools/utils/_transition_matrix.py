
import ABCParse
import anndata
import numpy as np
import scipy.sparse
import warnings

warnings.filterwarnings("ignore")

class TransitionMatrix(ABCParse.ABCParse):
    """
    pre-requisite: run `tl.velocity_graph` to compute cosine correlations
    """

    def __init__(
        self,
        self_transitions: bool = True,
        use_negative_cosines: bool = True,
        scale: float = 10,
        *args,
        **kwargs,
    ):
        self.__parse__(locals())

    @property
    def _velocity_graph_key(self):
        return f"{self._velocity_key}_graph"

    @property
    def _velocity_graph_neg_key(self):
        return f"{self._velocity_key}_graph_neg"

    @property
    def _has_velocity_graph(self):
        return self._velocity_graph_key in self._adata.uns

    @property
    def graph(self):
        if not hasattr(self, "_graph"):
            self._graph = self._adata.obsp[self._velocity_graph_key]
        return self._graph

    @property
    def graph_negative(self):
        if not hasattr(self, "_graph_neg"):
            self._graph_neg = self._adata.obsp[self._velocity_graph_neg_key]
        return self._graph_neg

    def compute_self_transitions(self, graph):
        confidence = graph.max(1).A.flatten()
        self_prob = np.clip(np.percentile(confidence, 98) - confidence, 0, 1)
        graph.setdiag(self_prob)
        return graph
    
    def _normalize(self, input):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if scipy.sparse.issparse(input):
                return input.multiply(scipy.sparse.csr_matrix(1.0 / np.abs(input).sum(1)))
            return input / input.sum(1)

    @property
    def auto_densify(self):
        return self._adata.n_obs < 1e4

    def __call__(
        self,
        adata: anndata.AnnData,
        velocity_key: str = "velocity",
        force_dense: bool = False,
        disable_dense: bool = False,
        *args,
        **kwargs,
    ) -> scipy.sparse.csr_matrix:
        """
        Args:
            adata (anndata.AnnData): description.
            
            velocity_key (str): description. **Default** = "velocity"
            
        Returns:
            T (scipy.sparse.csr_matrix): transition matrix.
        """
        
        self.__update__(locals())

        graph = self.compute_self_transitions(self.graph)
        T = np.expm1(graph * self._scale)

        if self._use_negative_cosines:
            T -= np.expm1(-self.graph_negative * self._scale)
        else:
            T += np.expm1(self.graph_negative * self._scale)
            T.data += 1

        T.setdiag(0)
        T.eliminate_zeros()
        
        T = self._normalize(T)
        
        assert sum([self._force_dense, self._disable_dense]) <= 1, "pick one"
        
        if self._force_dense or self.auto_densify and not self._disable_dense:
                T = T.A
        return T


# -- API-facing function: ----------
def transition_matrix(
    adata: anndata.AnnData,
    velocity_key: str = "velocity",
    self_transitions: bool = True,
    use_negative_cosines: bool = True,
    scale: float = 10,
    force_dense: bool = False,
    disable_dense: bool = False,
    return_cls: bool = False,
) -> scipy.sparse.csr_matrix:
    """Compute transition matrix"""
    
    _transition_matrix = TransitionMatrix(
        self_transitions=self_transitions,
        use_negative_cosines=use_negative_cosines,
        scale=scale,
    )
    T = _transition_matrix(adata=adata, velocity_key=velocity_key, force_dense=force_dense, disable_dense=disable_dense)
    
    if return_cls:
        return _transition_matrix, T
    return T