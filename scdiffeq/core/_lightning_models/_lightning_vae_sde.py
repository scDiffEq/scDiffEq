
# -- import packages: ----------------------------------------------------------
import torch


# -- import local dependencies: ------------------------------------------------
from .base_models import BaseLightningSDE
from .mix_ins import VAEPreTrainMixIn, VAEMixIn
from ._sinkhorn_divergence import SinkhornDivergence
from ..utils import sum_normalize


NoneType = type(None)


class LightningVAESDE(VAEPreTrainMixIn, VAEMixIn, BaseLightningSDE):
    def __init__(
        self,
        func,
        pretrain_epochs=20,
        dt=0.1,
        pretrain_lr=1e-2,
        train_lr=1e-5,
        pretrain_step_size=200,
        train_step_size=50,
        optimizers=[torch.optim.RMSprop, torch.optim.RMSprop],
        lr_schedulers=[
            torch.optim.lr_scheduler.StepLR,
            torch.optim.lr_scheduler.StepLR,
        ],
    ):
        super(LightningVAESDE, self).__init__()

        self.__parse__(kwargs=locals())

        self.func = func

        self.SinkhornLoss = SinkhornDivergence(backend="auto")
        self.MSELoss = torch.nn.MSELoss(reduction="sum")

        VAE_params = list(self.Encoder.parameters()) + list(self.Decoder.parameters())
        self.optimizers = [
            optimizers[0](VAE_params, lr=pretrain_lr),
            optimizers[1](self.parameters(), lr=train_lr),
        ]
        self.lr_schedulers = [
            lr_schedulers[0](self.optimizers[0], step_size=pretrain_step_size),
            lr_schedulers[1](self.optimizers[1], step_size=train_step_size),
        ]
        self.hparams["func_type"] = "NeuralSDE"
        self.hparams["func_description"] = str(self.func)
        self.save_hyperparameters(ignore=["func", "optimizer", "lr_scheduler"])

    def process_batch(self, batch):
        """called in step"""

        t = batch[0].unique()
        X = batch[1].transpose(1, 0)
        X0 = X[0]

        return X, X0, t

    def loss(self, X, X_hat):
        return self.SinkhornLoss(X.contiguous(), X_hat.contiguous())

    def record_loss(self, loss, stage):
        """Record loss. called in step"""

        log_msg = "{}"
        if not isinstance(stage, NoneType):
            log_msg = f"{stage}_" + "{}"
        for i, l in enumerate(loss):
            self.log(log_msg.format(i), l.item())

    def step(self, batch, batch_idx, stage):

        X, X0, t = self.process_batch(batch)
        X_hat = self.forward(X0, t)
        loss = self.loss(X, X_hat)
        self.record_loss(loss, stage)
        return loss.sum()

    def training_step(self, batch, batch_idx, optimizer_idx):

        if self.pretrain_criteria(optimizer_idx):
            return self.pretrain_step(batch, batch_idx, stage="pretraining_train")
        elif self.train_criteria(optimizer_idx):
            return self.step(batch, batch_idx, stage="training")

    def validation_step(self, batch, batch_idx, optimizer_idx):

        if self.pretrain_criteria(optimizer_idx):
            return self.pretrain_step(batch, batch_idx, stage="pretraining_validation")
        elif self.train_criteria(optimizer_idx):
            return self.step(batch, batch_idx, stage="validation")
