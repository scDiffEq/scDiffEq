
# -- import packages: ----------------------------------------------------------
import torch


# -- import local dependencies: ------------------------------------------------
from .base_models import BaseLightningSDE
from .mix_ins import PotentialMixIn
from ._sinkhorn_divergence import SinkhornDivergence
from ..utils import sum_normalize


# -- set typing: ---------------------------------------------------------------
NoneType = type(None)


# -- model class: --------------------------------------------------------------
class LightningSDE_LatentPotential(PotentialMixIn, BaseLightningSDE):
        
    def __init__(
        self,
        func,
        dt=0.1,
        lr=1e-5,
        logqp=True,
        step_size=10,
        optimizer=torch.optim.RMSprop,
        lr_scheduler=torch.optim.lr_scheduler.StepLR,
    ):
        super().__init__()
        
        self.func = func
        self.loss_func = SinkhornDivergence()
        self.optimizers = [optimizer(self.parameters(), lr=lr)]
        self.lr_schedulers = [lr_scheduler(self.optimizers[0], step_size=step_size)]
        self.hparams['func_type'] = "NeuralSDE"
        self.hparams['func_description'] = str(self.func)
        self.save_hyperparameters(ignore=["func"])

    def process_batch(self, batch):
        """called in step"""

        t = batch[0].unique()
        X = batch[1].transpose(1, 0)
        X0 = X[0]
        W_hat = sum_normalize(batch[2].transpose(1, 0))

        return X, X0, W_hat, t

    def loss(self, X, X_hat, W, W_hat):
        X, X_hat = X.contiguous(), X_hat.contiguous()
        W, W_hat = W.contiguous(), W_hat.contiguous()
        return self.loss_func(W, X, W_hat, X_hat)

        
    def forward(self, X0, t, **kwargs):
        """
        We want this to be easily-accesible from the outside, so we
        directly define the forward step with the integrator code.
        """
                
        return self.integrate(
            X0=X0,
            t=t,
            dt=self.hparams['dt'],
            logqp=self.hparams['logqp'],
            **kwargs,
        )
    
    def record(self, loss, stage):
        """Record loss. called in step"""
        
        log_msg = "{}"
        if not isinstance(stage, NoneType):
            log_msg = f"{stage}_sinkhorn_" + "{}"
        for i, l in enumerate(loss):
            self.log(log_msg.format(i), l.item())
            
    def step(self, batch, batch_idx, stage=None):
        """Batch should be from a torch DataLoader"""
        
        X, X0, W_hat, t = self.process_batch(batch)
        if stage == "predict":
            t = self.t
        X_hat, kl_div = self.forward(X0, t)
        if stage == "predict":
            return X_hat
        else:
            W = sum_normalize(torch.ones_like(W_hat))
            loss = self.loss(X, X_hat, W, W_hat)
            self.record(loss, stage)
            self.log(f"{stage}_kl_div", kl_div.sum().item())
            return loss.sum() + kl_div.sum()
        
#     def validation_step(self, batch, batch_idx):
#         torch.set_grad_enabled(True)
#         return self.step(batch, batch_idx, stage="validation")
    