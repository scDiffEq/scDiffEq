
# -- import packages: ------------------------------------------------------------------
import torch
import numpy as np

# -- import local dependencies: --------------------------------------------------------
from ._function_credentials import Credentials
from ..utils import extract_func_kwargs, Base
from ._loss_log import LossLog
from ._batch import Batch


# -- supporting classes / functions: ---------------------------------------------------

loss_log = LossLog()


class UniversalForwardIntegrator(Base):
    def __init__(self, func, adjoint=False):

        self.func = func
        creds = Credentials(self.func, adjoint=adjoint)
        self.func_type, self.mu_is_potential, self.sigma_is_potential = creds()
        self.integrator = creds.integrator

    def __call__(self, X0, t, dt=0.1):

        self.__parse__(locals(), ignore=["self", "X0", "t"])
        int_kwargs = extract_func_kwargs(func=self.integrator, kwargs=self._KWARGS)
        return self.integrator(self.func, X0, t, **int_kwargs)


# -- universal forward: ----------------------------------------------------------------
def forward(self, batch, stage=None, t=None, expand=False):
    
    if not isinstance(t, torch.Tensor):
        t = self.t
        
    _forward = UniversalForwardIntegrator(self.func, adjoint=self.adjoint)
    func_type = _forward.func_type
        
    batch = Batch(batch, stage, func_type=func_type, t=t, expand=expand)
    X_hat = _forward(X0=batch.X0, t=batch.t, dt=self.dt)
            
        
    if not stage in ['predict']:
        x = batch.X[1:].flatten(0, 1)
        shuffle_idx = np.random.choice(range(len(x)), size=len(x), replace=False)
        
        if not self.real_time:
            loss = self.loss_func(
                batch.X[1:].flatten(0, 1)[shuffle_idx][None, :, :],
                X_hat[1:].flatten(0, 1)[None, :, :],
                batch.W[1:].flatten(0, 1)[None, :, :].contiguous(),
                batch.W[1:].flatten(0, 1)[None, :, :].contiguous(),
            )
            loss_log.positional(self, loss, stage, batch)

            if not stage in ["predict", "test"]:
                for i, t in enumerate(batch.t[:1]):
                    log_ = "{}_loss_{}".format(stage, str(int(i)))
                    val_ = loss["positional"][i].item()
                    self.log(log_, val_)
        else:
            loss = self.loss_func(
                batch.X.contiguous(),
                X_hat.contiguous(),
                batch.W.contiguous(),
                batch.W.contiguous(),
            )

            loss_log.positional(self, loss, stage, batch)

            if not stage in ["predict", "test"]:
                for i, t in enumerate(batch.t):
                    log_ = "{}_loss_{}".format(stage, str(int(i)))
                    val_ = loss["positional"][i].item()
                    self.log(log_, val_)

        return {
            "loss": loss["positional"].sum(),
            "pred": X_hat,
            "batch": batch,
               }
    
    return {
            "pred": X_hat,
            "batch": batch,
               }

