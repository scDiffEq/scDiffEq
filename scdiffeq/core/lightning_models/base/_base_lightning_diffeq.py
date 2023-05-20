

# -- import packages: ----------------------------------------------------------
import lightning
import torchsde
from abc import abstractmethod


# -- import local dependencies: ------------------------------------------------
from ._batch_processor import BatchProcessor
from ._sinkhorn_divergence import SinkhornDivergence

from ... import utils


# -- DiffEq class: -------------------------------------------------------------
class BaseLightningDiffEq(lightning.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.sinkhorn_divergence = SinkhornDivergence(backend="auto", **kwargs)
        self.process_batch = BatchProcessor
        self.COMPLETED_EPOCHS = 0
        
    def _update_lit_diffeq_hparams(self, model_params):
        for key, val in self.hparams.items():
            if key in model_params.keys():
                if val != model_params[key]:
                    self.hparams.update({key: model_params[key]})
        
    # -- setup: ----------------------------------------------------------------
    def _configure_optimizers_schedulers(self):
        """Assumes no pre-train - i.e., a single optimizer, scheduler"""
        
        optimizer = self.hparams['train_optimizer']
        scheduler = self.hparams['train_scheduler']

        self._optimizers = [optimizer(self.parameters(), lr=self.hparams['train_lr'])]
        self._schedulers = [
            scheduler(
                optimizer=self._optimizers[0],
                step_size=self.hparams['train_step_size']),
        ]

    def _configure_torch_modules(self, func, kwargs):
        
        kwargs['state_size'] = self.hparams['latent_dim']
        self.DiffEq = func(**utils.function_kwargs(func, kwargs))
        
    @property
    def PRETRAIN(self):
        return False
    
    # -- integrator stuff: -----------------------------------------------------
    @property
    def _INTEGRATOR(self):
        if self.hparams["adjoint"]:
            return torchsde.sdeint_adjoint
        return torchsde.sdeint

    def integrate(self, Z0, t, dt, logqp, **kwargs):
        return self._INTEGRATOR(
            sde=self.DiffEq,
            y0=Z0,
            ts=t,
            dt=dt,
            logqp=logqp,
            **kwargs,
        )

    # -- sinkhorn stuff: -------------------------------------------------------
    def compute_sinkhorn_divergence(self, X, X_hat, W, W_hat):
        return self.sinkhorn_divergence(
            W.contiguous(), X.contiguous(), W_hat.contiguous(), X_hat.contiguous()
        ).requires_grad_()

    def log_sinkhorn_divergence(self, sinkhorn_loss, t, stage):
        for i in range(len(t)):
            _t = round(t[i].item(), 3)
            msg = f"sinkhorn_{_t}_{stage}"
            val = sinkhorn_loss[i]
            self.log(msg, val)

        return sinkhorn_loss.sum()
    
    def log_lr(self):
        
        for i, opt in enumerate(self.optimizers()):
            for j, pg in enumerate(opt.optimizer.state_dict()["param_groups"]):
                self.log(f"opt_{i}_param_group_{j}_lr", pg["lr"])

    # -- custom steps: -------------------------------------------------------------
    @abstractmethod
    def forward(self, Z0, t, **kwargs):
        """most likely over-written in another class"""
        ...

    @abstractmethod
    def step(self, batch, batch_idx, stage=None):
        print("WARNING: The base (empty) step is being called from `_base_lightning_diffeq.py`")
        ...

    # -- LightningModule methods: ----------------------------------------------
    def training_step(self, batch, batch_idx, *args, **kwargs):
        return self.step(batch, batch_idx, stage="training")

    def validation_step(self, batch, batch_idx=None, *args, **kwargs):
        return self.step(batch, batch_idx, stage="validation")

    def test_step(self, batch, batch_idx=None, *args, **kwargs):
        return self.step(batch, batch_idx, stage="test")

    def predict_step(self, batch, batch_idx=None, *args, **kwargs):
        return self.step(batch, batch_idx, stage="predict")

    def configure_optimizers(self):
        return self._optimizers, self._schedulers
