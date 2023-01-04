
# -- import packages: ------------------------------------------------------------------
import torch


# -- import local dependencies: --------------------------------------------------------
from ._batch import Batch
from ._universal_forward_integrator import UniversalForwardIntegrator
from ._loss_log import LossLog

loss_log = LossLog()

# -- universal forward: ----------------------------------------------------------------
def forward(self, batch, stage=None, t=None, expand=False):
    
    if not isinstance(t, torch.Tensor):
        t = self.t
        
    forward = UniversalForwardIntegrator(self.func)
    
    func_type = forward.func_type
        
    batch = Batch(batch, stage, func_type=func_type, t=t, expand=expand)
    X_hat = forward(X0=batch.X0, t=batch.t, dt=self.dt)
    if not stage in ['predict']:
        loss = self.loss_func(
            batch.X.contiguous(),
            X_hat.contiguous(),
            batch.W.contiguous(),
            batch.W.contiguous(),
        )

        loss_log.positional(self, loss, stage, batch)

        if not stage in ["predict", "test"]:
            for i, t in enumerate(batch.t): # ["ts"]
                log_ = "{}_loss_{}".format(stage, str(int(t)))
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

