import ABCParse
from typing import List
import torch

from ... import utils

NoneType = type(None)

class BatchRepr:
    def __init__(self, batch):

        self.header = "🍪 LitDataBatch"
        self.X = f"- X: {list(batch.X.shape)}"
        self.W = f"- W: {list(batch.W.shape)}"
        self.W_hat = f"- W_hat: {list(batch.W_hat.shape)}"
        if not isinstance(batch.F_idx, NoneType):
            self.F_idx = f"- F_idx: {list(batch.F_idx.shape)}"
        else:
            self.F_idx = ""

    def __call__(self):
        return "\n".join([self.header, self.X, self.W, self.W_hat, self.F_idx])
    

class BatchProcessor(ABCParse.ABCParse):
    """
    Batch Items:
    ------------
    0. t - time. always required
    1. X - data. tensor of size: [batch_size, len(t), n_dim]. always required.
    2. W - weight. tensor of size: [batch_size, len(t), 1]
    3. F_idx - fate_idx. tensor of size: [batch_Size, len(t), 1]
    """
    def __init__(self, batch: List[torch.Tensor], batch_idx: int)->None:
        
        self.__parse__(locals(), private=['batch'])
        
    @property
    def device(self):
        return self._batch[0].device
        
    @property
    def n_batch_items(self)->int:
        return len(self._batch)
    
    @property
    def batch_size(self)->int:
        return self._batch[0].shape[0]
    
    @property
    def t(self)->torch.Tensor:
        return self._batch[0].unique()
    
    @property
    def X(self)->torch.Tensor:
        return self._batch[1].transpose(1, 0).contiguous()
    
    @property
    def X0(self)->torch.Tensor:
        return self.X[0]
    
    @property
    def W_hat(self)->torch.Tensor:
        if self.n_batch_items >= 3:
            W = self._batch[2].transpose(1, 0)
            return utils.sum_normalize(W, sample_axis=1).contiguous()
        return utils.sum_normalize(torch.ones([len(self.t), self.batch_size, 1], device=self.device)).contiguous()
        
    @property
    def W(self)->torch.Tensor:
        return utils.sum_normalize(torch.ones_like(self.W_hat)).contiguous()
    
    @property
    def F_idx(self):
        if self.n_batch_items >= 4:
            return (
                self._batch[3]
                .transpose(1, 0)[0]
                .detach()
                .cpu()
                .numpy()
                .flatten()
                .astype(int)
                .astype(str)
            )

    def __repr__(self)->str:
        return BatchRepr(self)()
