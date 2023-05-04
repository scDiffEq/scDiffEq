
from ._pre_train_mix_in import PreTrainMixIn

class VAEPreTrainMixIn(PreTrainMixIn):
    """This gives us the flexibility to pre-train the VAE before fitting an SDE."""

    def __init__(self, *args, **kwargs):
        super(VAEPreTrainMixIn, self).__init__()

    def pretrain_loss(self, X, X_hat):
        return self.MSELoss(X, X_hat)

    def record_pretrain_loss(self, loss, stage):
        self.log(f"{stage}_loss", loss.item())

    def pretrain_step(self, batch, batch_idx, stage):
        """"""
        X, X0, t = self.process_batch(batch)
        X_hat = self.decode(self.encode(X))
        loss = self.pretrain_loss(X, X_hat)
        self.record_pretrain_loss(loss, stage)

        return loss.sum()

    def pretrain_criteria(self, optimizer_idx):
        return self.current_epoch < self.pretrain_epochs and optimizer_idx == 0

    def train_criteria(self, optimizer_idx):
        return self.current_epoch >= self.pretrain_epochs and optimizer_idx == 1