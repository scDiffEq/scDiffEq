
from ._batch import Batch
from torchsde import sdeint

# -- default SDE forward: --------------------------------------------------------------
def SDE_forward(self, batch, stage=None):

    batch = Batch(batch, func_type="neural_SDE")
    X_hat = sdeint(self.func, batch.X0, **batch.t, **{"dt": 0.1})
    
    X = batch.X.contiguous()
    X_hat = X_hat.contiguous()
    W = batch.W.contiguous()
    W_hat = batch.W.contiguous()

    loss = self.loss_func(
        batch.X.contiguous(),
        X_hat.contiguous(),
        batch.W.contiguous(),
        batch.W.contiguous(),
    )
    return {"loss": loss["positional"].sum()}