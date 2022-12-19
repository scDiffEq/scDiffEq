
__module_name__ = "_batch_forward.py"
__doc__ = """To-do."""
__version__ = "0.0.44"
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# -- import packages: --------------------------------------------------------------------
from abc import ABC, abstractmethod
from pytorch_lightning import LightningModule, loggers
from neural_diffeqs import NeuralSDE, NeuralODE
from torch_nets import TorchNet
import torch


# -- import local dependencies: ----------------------------------------------------------
# from ..configs import func_params, extract_func_kwargs
# from ..configs._data_configure import InputConfiguration
from ..forward import credential_handoff


# -- Supporting functions: ---------------------------------------------------------------
def _sum_norm(x):
    return x / x.sum(1)[:, None]


# -- Main class: -------------------------------------------------------------------------
class BatchForward:
    """
    This class can be modified to add additional flexibility for other types of
    forward functions. i.e., +/- different or additional loss functions (e.g.,
    velo, fate) and/or dim. reduction (e.g., VAE)
    """
    def __init__(self, func, device="cuda:0"): # , loss_function, device):
        """TODO: add docs."""

        self.integrator, self._func_type = credential_handoff(func)
        self.func = func
        self.device = device
#         loss_function = loss_function(device)
#         parser(self, locals())

    @property
    def time_arg(self):
        if self._func_type == "neural_SDE":
            return {"ts": self.t}
        return {"t": batch[0].unique()}

    def __parse__(self, batch, ignore=["self", "ignore", "batch"]):

        t = batch[0].unique()
        X = batch[1].transpose(1, 0)
        X0 = X[0]
        W = batch[2].transpose(1, 0)

        if len(batch) == 4:
            W_hat = _sum_norm(batch[3].transpose(1, 0))
        else:
            W_hat = _sum_norm(torch.ones_like(W))

        for k, v in locals().items():
            if not k in ignore:
                setattr(self, k, v.to(self.device))

    def __inference__(self, **kwargs):
        """
        t or ts is by necessity included in **kwargs
        dt is also most easily handled by kwargs.
        """
        kwargs.update(self.time_arg)
        self.X_hat = self.integrator(self.func, self.X0, **kwargs)
        return self.X_hat

    def __positional_loss__(self):

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
    
    def __velocity_loss__(self):
        """
        There are two steps:
        (4a) Calculate empirical velocities based on the positions
        (4b) Calculate loss between observered and predicted velocities.
        """
        
        pass
        

    def __log__(self, model, stage, loss):
        for n, i in enumerate(range(len(self.t))[-len(loss) :]):
            model.log("{}_{}_loss".format(stage, self.t[i]), loss[n])

    def __call__(self, model, batch, stage, **kwargs):
        """
        By default, __call___ will run:
        (1) __parse__(batch)
        (2) __inference__()
        (3) __positional_loss__()
        (4) __velocity_loss__()
        (5) __log__()
        Finally, it returns the output of loss.
        """
        
        print("-- BATCH INFO --")
        print("# OF ITEMS IN BATCH: {}".format(len(batch)))
        print("t: {}".format(batch[0].unique()))
        print("X shape: {}".format(batch[1].shape))
        print("W shape: {}".format(batch[2].shape))
        print("v shape: {}".format(batch[3].shape))
        
        print("t device: {}".format(batch[0].device))
        print("X device: {}".format(batch[1].device))
        print("W device: {}".format(batch[2].device))
        print("v device: {}".format(batch[3].device))
        
        inference_kwargs = extract_func_kwargs(self.integrator, kwargs)
        self.__parse__(batch)
        X_hat = self.__inference__(**inference_kwargs)
        if stage == "predict":
            return X_hat
        loss = self.__positional_loss__()
        self.__log__(model, stage, loss)
        return loss.sum()
