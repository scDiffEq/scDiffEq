
# -- import packages: ----------------------------------------------------------
import torch


# -- import local dependencies: ------------------------------------------------
from ._base_lightning_sde import BaseLightningSDE
from ._sinkhorn_divergence import SinkhornDivergence


# -- model class: --------------------------------------------------------------
class LightningSDE(BaseLightningSDE):
    def __init__(self, func, dt=0.1):
        super(LightningSDE, self).__init__()

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
        t = batch[0].unique()

        return X, X0, t

    def loss(self, X, X_hat):
        return self.loss_func(X.contiguous(), X_hat.contiguous())

    def record(self, loss):
        """called in step"""
        
        for i, l in enumerate(loss):
            self.log(i, l.item())

    def step(self, batch, batch_idx, stage=None):
        """Batch should be from a torch DataLoader"""
        
        X, X0, t = self.process_batch(batch)
        X_hat = self.forward(X0, t)
        loss = self.loss(X, X_hat)
        self.record(loss)
        
        return loss.sum()
