
__module_name__ = "_BaseModel.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import pytorch_lightning
import torch


# import local dependencies #
# ------------------------- #
from . import _base_ancilliary as base


class BaseModel(pytorch_lightning.LightningModule):
    def __init__(self):
        super(BaseModel, self).__init__()

    def model_setup(
        self,
        dataset,
        func,
        log_path="./",
        learning_rates=[1e-3],
        optimizers=[torch.optim.Adam],
        schedulers=[torch.optim.lr_scheduler.StepLR],
        scheduler_kwargs=[{"step_size": 100, "gamma": 0.9}],
        logger=pytorch_lightning.loggers.CSVLogger,
        flush_logs_every_n_steps=1,
        logger_kwargs={},
        dt=0.1,
        alpha=0.5,
        time_scale=None,
        loss_function=None,
    ):

        """
        this is where we define all the options such as what forward function we should use, etc.
        """

        self.dataset = dataset
        self.func = func
        self._learning_rates = learning_rates
        self._optimizers = optimizers
        self._schedulers = schedulers
        self._scheduler_kwargs = scheduler_kwargs
        self._time_scale = time_scale
        self._alpha = alpha
        self._dt = dt
        self.forward_function = base.ForwardFunctions(
            self.func,
            self._time_scale,
            self._alpha,
            self._dt,
            self.device,
        )

        self._log_path = log_path
        self._flush_logs_every_n_steps = flush_logs_every_n_steps
        self._logger = logger(
            save_dir=self._log_path,
            flush_logs_every_n_steps=self._flush_logs_every_n_steps,
            **logger_kwargs,
        )

        if not loss_function:
            self.loss_function = base.SinkhornDivergence()
        else:
            self.loss_function = loss_function

    def forward(self, X, t):

        X0, X_obs, W_obs = base.format_batched_inputs(X, t)
        X_hat = self.forward_function.step(self.func, X0=X0, t=t)
        W_hat = torch.ones_like(W_obs, requires_grad=True).to(self.device)
        sinkhorn_loss = self.loss_function.compute(W_hat, X_hat, W_obs, X_obs, t)

        return {"X_hat": X_hat, "sinkhorn_loss": sinkhorn_loss, "t": t}

    def training_step(self, batch, batch_idx):

        FWD = self.forward(X=batch, t=self.dataset._train_time)
        base.update_loss_logs(
            self,
            FWD["sinkhorn_loss"],
            FWD["t"],
            self.current_epoch,
            batch_idx,
            "train",
            "sinkhorn",
            "d",
        )

        return FWD["sinkhorn_loss"].sum()

    def validation_step(self, batch, batch_idx):

        base.retain_gradients(self)
        FWD = self.forward(X=batch, t=self.dataset._train_time)
        base.update_loss_logs(
            self,
            FWD["sinkhorn_loss"],
            FWD["t"],
            self.current_epoch,
            batch_idx,
            "val",
            "sinkhorn",
            "d",
        )

        return FWD["sinkhorn_loss"]

    def test_step(self, batch, batch_idx):

        base.retain_gradients(self)
        FWD = self.forward(X=batch, t=self.dataset._test_time)
        base.update_loss_logs(
            self,
            FWD["sinkhorn_loss"],
            FWD["t"],
            self.current_epoch,
            batch_idx,
            "test",
            "sinkhorn",
            "d",
        )

        return FWD["sinkhorn_loss"]

    def configure_optimizers(self):
        return base.optim_config(
            param_groups=[self.parameters()],
            learning_rates=self._learning_rates,
            optimizers=self._optimizers,
            schedulers=self._schedulers,
            scheduler_kwargs=self._scheduler_kwargs,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        FWD["sinkhorn_loss"] = self.forward(X=batch, t=self.dataset._test_time)
        base.update_loss_logs(
            self,
            FWD["sinkhorn_loss"],
            FWD["t"],
            self.current_epoch,
            batch_idx,
            "predict",
            "sinkhorn",
            "d",
        )
        return FWD