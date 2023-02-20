
# -- import packages: ----------------------------------------------------------
import torch


# -- import local dependencies: ------------------------------------------------
from ._base_lightning_diffeqs import BaseLightningSDE
from ._sinkhorn_divergence import SinkhornDivergence
from ._potential_mixin import PotentialMixIn

from ..utils import sum_normalize

# -- model class: --------------------------------------------------------------
class LightningSDE(BaseLightningSDE):
    def __init__(self, func, dt=0.1, lr=1e-4):
        super(LightningSDE, self).__init__()

        self.func = func
        self.loss_func = SinkhornDivergence()
        self.optimizers = [torch.optim.RMSprop(self.parameters(), lr=lr)]       # TODO: replace
        self.lr_schedulers = [
            torch.optim.lr_scheduler.StepLR(self.optimizers[0], step_size=10)   # TODO: replace
        ]
        self.save_hyperparameters(ignore=["func"])

    def process_batch(self, batch):
        """called in step"""
        
        X = batch[1].transpose(1, 0)
        X0 = X[0]
        t = batch[0].unique()
        W = batch[2].transpose(1, 0)
        W = sum_normalize(W)

        return X, W, X0, t

    def loss(self, X, X_hat, W, W_hat):
        X, X_hat = X.contiguous(), X_hat.contiguous()
        W, W_hat = W.contiguous(), W_hat.contiguous()
        return self.loss_func(W, X, W_hat, X_hat)

    def step(self, batch, batch_idx, stage=None):
        """Batch should be from a torch DataLoader"""
        
        X, W, X0, t = self.process_batch(batch)
        if stage == "predict":
            t = self.t
        X_hat = self.forward(X0, t)
        if stage == "predict":
            return X_hat
        else:
            W_hat = sum_normalize(torch.ones_like(W))
            loss = self.loss(X, X_hat, W, W_hat)
            self.record(loss, stage)
            return loss.sum()
            

class LightningPotentialSDE(LightningSDE, PotentialMixIn):
    def __init__(self, func, dt=0.1, lr=1e-4):
        super(LightningPotentialSDE, self).__init__(func, dt=0.1, lr=1e-4)
