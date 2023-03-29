
# -- import packages: ----------------------------------------------------------
import torch


# -- import local dependencies: ------------------------------------------------
from .base_models import BaseLightningODE
from .mix_ins import PotentialMixIn
from ._sinkhorn_divergence import SinkhornDivergence


def min_max_norm(t: torch.Tensor)->torch.Tensor:
    return (t - t.min()) / (t.max() - t.min()) * 1e-3 # TODO: make param


# -- model class: --------------------------------------------------------------
class LightningODE(BaseLightningODE):
    def __init__(self, func):
        super(LightningODE, self).__init__()

        self.func = func
        self.loss_func = SinkhornDivergence()
        self.optimizers = [torch.optim.RMSprop(self.parameters())]              # TODO: replace
        self.lr_schedulers = [
            torch.optim.lr_scheduler.StepLR(self.optimizers[0], step_size=10)   # TODO: replace
        ]
        self.save_hyperparameters(ignore=["func"])    

    def process_batch(self, batch):
        """called in step"""
        
        X = batch[1].transpose(1, 0)
        X0 = X[0]
        t = min_max_norm(batch[0].unique())

        return X, X0, t

    def loss(self, X, X_hat):
        return self.loss_func(X.contiguous(), X_hat.contiguous())

    def step(self, batch, batch_idx, stage=None):
        """Batch should be from a torch DataLoader"""
        
        X, X0, t = self.process_batch(batch)
        X_hat = self.forward(X0, t)
        loss = self.loss(X, X_hat)
        if stage in ["training", "validation"]:
            self.record(loss, stage)
            
        return loss.sum()

class LightningPotentialODE(LightningODE, PotentialMixIn):
    def __init__(self, func):
        super(LightningPotentialODE, self).__init__(func)
