
# -- import packages: ----------------------------------------------------------
import torch


# -- import local dependencies: ------------------------------------------------
from ._base_lightning_diffeqs import BaseLightningDriftNet
from ._sinkhorn_divergence import SinkhornDivergence
from ._potential_mixin import PotentialMixIn

# -- model class: --------------------------------------------------------------
class LightningDriftNet(BaseLightningDriftNet):
    def __init__(self, func, dt=0.1, stdev=torch.Tensor([0.5])):
        super(LightningDriftNet, self).__init__()

        self.func = func
        self.loss_func = SinkhornDivergence()
        self.optimizers = [torch.optim.RMSprop(self.parameters())]  # TODO: replace
        self.lr_schedulers = [
            torch.optim.lr_scheduler.StepLR(
                self.optimizers[0], step_size=10
            )  # TODO: replace
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

    
    def step(self, batch, batch_idx, stage=None):
        """Batch should be from a torch DataLoader"""

        X, X0, t = self.process_batch(batch)
        X_hat = self.forward(X0, t)
        loss = self.loss(X, X_hat)
        if stage in ["training", "validation"]:
            self.record(loss, stage)

        return loss.sum()
    

class LightningPotentialDriftNet(LightningDriftNet, PotentialMixIn):
    def __init__(self, func):
        super(LightningPotentialDriftNet, self).__init__(func)
