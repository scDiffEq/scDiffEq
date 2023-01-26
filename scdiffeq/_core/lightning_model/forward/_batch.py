

# -- import packages: ------------------------------------------------------------------
import torch


# -- import local dependencies: --------------------------------------------------------
from ...utils import sum_normalize, AutoParseBase


# -- supporting functions: -------------------------------------------------------------
def _expanded_X0(batch, N=2000):
    return torch.stack(
        [self.X0[i].expand(N, 50) for i in range(len(self.X0))]
    ).reshape(-1, 50)


# -- Model-facing Batch: ---------------------------------------------------------------
class Batch(AutoParseBase):
    """Catch batch and make sure it's in the right format."""    
    def __init__(self, batch, stage, func_type, t=None, expand=False):
        self.__parse__(locals(), public=[None])
        
    def _expand_X0(self, N=2000):
        n_unique, n_dim = self.X.shape[1], self.X.shape[2]        
        return self.X.expand(N, n_unique, n_dim).reshape(n_unique * N, n_dim)
    
    @property
    def func_type(self):
        return self._func_type
    
    @property
    def stage(self):
        return self._stage
    
    @property
    def t(self):
        if not isinstance(self._t, torch.Tensor):
            self._t = self._batch[0].unique()
        return self._t
    
    @property
    def X(self):
        return self._batch[1].transpose(1,0)

    @property
    def X0(self):
        if self._expand:
            return self._expand_X0(N=self._expand)
        return self.X[0]

    @property
    def W(self):
        return sum_normalize(self._batch[2].transpose(1,0))
    
    @property
    def cell_idx(self):
        return self._batch[3].transpose(1,0).cpu().numpy().astype(int).astype(str)
    
    @property
    def V(self):
        if len(self._batch) >= 5:
            return self._batch[4].transpose(1,0)
        