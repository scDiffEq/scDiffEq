
__module_name__ = "_base_model.py"
__doc__ = """To-do."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# specify version: -----------------------------------------------------------------------
__version__ = "0.0.44"


# -- import packages: --------------------------------------------------------------------
from abc import ABC, abstractmethod

from pytorch_lightning import LightningModule, loggers
from neural_diffeqs import NeuralSDE, NeuralODE
from torch_composer import TorchNet
import torch


# -- import local dependencies: ----------------------------------------------------------
from ._base_utility_functions import extract_func_kwargs
from ._sinkhorn_divergence import SinkhornDivergence
from ._integrators import credential_handoff
from ._configure import InputConfiguration


class BaseBatchForward(ABC):
    def __init__(self, func, loss_function, device):
        """To-do: add docs."""

        self.integrator, self.func_type = credential_handoff(func)
        self.loss_function = loss_function(device)
        self.func = func

    @abstractmethod
    def __parse__(self):
        pass

    @abstractmethod
    def __inference__(self):
        pass

    @abstractmethod
    def __loss__(self):
        pass
    
    @abstractmethod
    def __call__(self, model, batch, stage, **kwargs):
        pass


class BatchForward(BaseBatchForward):
    """
    Subsituting this class, subclassed from the `BaseBatchForward` module, above
    adds flexibility for other types of forward functions. i.e., +/- different or
    additional loss functions (e.g., velo, fate) and/or dim. reduction (e.g., VAE)
    """
    def _sum_norm(self, W):
        return W / W.sum(1)[:, None]

    def _format_sinkhorn_weights(self, W, W_hat):
        self.W, self.W_hat = self._sum_norm(W), self._sum_norm(W_hat)

    def _format_t(self, batch):
        self.t = batch[0].unique()

        if self.func_type == "neural_SDE":
            self.t_arg = {"ts": self.t}
        else:
            self.t_arg = {"t": batch[0].unique()}

    def __parse__(self, batch):

        self._format_t(batch)

        if len(batch) >= 3:
            W = batch[2].transpose(1, 0)

        if len(batch) == 4:
            W_hat = batch[3].transpose(1, 0)
            self._format_sinkhorn_weights(W, W_hat)

        self.X = batch[1].transpose(1, 0)
        self.X0 = self.X[0]

    def __inference__(self, **kwargs):
        """
        t or ts is by necessity included in **kwargs
        dt is also most easily handled by kwargs.
        """
        kwargs.update(self.t_arg)
        self.X_hat = self.integrator(self.func, self.X0, **kwargs)
        return self.X_hat

    def __loss__(self):

        if self.X_hat.shape[0] > len(self.t):
            time_slice = torch.linspace(0, (self.X_hat.shape[0] - 1), len(self.t)).to(
                int
            )
            X_hat = self.X_hat[time_slice.to(int)].contiguous()
        else:
            X_hat = self.X_hat.contiguous()

        return self.loss_function(
            X_hat, self.X.contiguous(), self.W.contiguous(), self.W_hat.contiguous()
        )

    def __log__(self, model, stage, loss):
        for n, i in enumerate(range(len(self.t))[-len(loss):]):
            model.log("{}_{}_loss".format(stage, self.t[i]), loss[n])

    def __call__(self, model, batch, stage, **kwargs):
        """
        By default, __call___ will run:
        (1) __parse__(batch)
        (2) __inference__()
        (3) __loss__()
        (4) __log__()
        Finally, it returns the output of loss.
        """
        inference_kwargs = extract_func_kwargs(self.integrator, kwargs)
        self.__parse__(batch)
        X_hat = self.__inference__(**inference_kwargs)
        if stage == "predict":
            return X_hat
        loss  = self.__loss__()
        self.__log__(model, stage, loss)
        return loss.sum()

# ----------------------------------------------------------------------------------------

def set_key_as_attr(self, k, v, hide):

    if hide == "all":
        setattr(self, "_{}".format(k), v)

    elif isinstance(hide, str):
        if k == hide:
            setattr(self, "_{}".format(k), v)

    elif isinstance(hide, list):
        if k in hide:
            setattr(self, "_{}".format(k), v)

    else:
        setattr(self, k, v)
        
# -- Lightning base: ---------------------------------------------------------------------
class BaseLightningModel(LightningModule):
    """Base pytorch-lightning model wrapped / trained within models.scDiffEq"""
    
    def __parse__(self, kwargs, hide=None, ignore=["self", "__class__"]):
        
        for k, v in kwargs.items():
            if not k in ignore:
                set_key_as_attr(self, k, v, hide)
    
    def __configure_forward_step__(self, ignore_t0=True):
        """To-Do: docs"""
        forward_step = BatchForward(self.func,
                                    loss_function = SinkhornDivergence,
                                    device = self.device,
                                   )
        setattr(self, "forward", getattr(forward_step, "__call__"))
        setattr(self, "integrator", getattr(forward_step, "integrator"))
        setattr(self, "func_type", getattr(forward_step, "func_type"))

    def configure_optimizers(self):
        """To-Do: docs"""
        # TODO: add optimizer / scheduler config to input args through sdq.models.scDiffEq
        optimizer = torch.optim.RMSprop(
            self.parameters(), **self.optimizer_kwargs # .hparams["optimizer_kwargs"]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, **self.scheduler_kwargs # .hparams[""]
        )
        return [optimizer], [scheduler]


# LightningModel: ------------------------------------------------------------------------
class LightningModel(BaseLightningModel):
    """TODO: docs"""
    def __init__(self,
                 func: [NeuralSDE, NeuralODE, TorchNet] = None,
                 dt: float = 0.1,
                 optimizer_kwargs={},
                 scheduler_kwargs={},
                 **kwargs,
                ):
        """TODO: docs"""
        super(LightningModel, self).__init__()
        self.__parse__(locals())
        self.__configure_forward_step__()
        
    def training_step(self, batch, batch_idx):
        return self.forward(self, batch, stage="train", dt=self.dt)

    def validation_step(self, batch, batch_idx):
        return self.forward(self, batch, stage="val", dt=self.dt)

    def test_step(self, batch, batch_idx):
        return self.forward(self, batch, stage="test", dt=self.dt)

    def predict_step(self, batch, batch_idx):
        return self.forward(self, batch, stage="predict", dt=self.dt)
