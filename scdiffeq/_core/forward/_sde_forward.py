
# -- import packages: ------------------------------------------------------------------
from torchsde import sdeint


# -- import local dependencies: --------------------------------------------------------
from ._batch import Batch


# -- default SDE forward: --------------------------------------------------------------
def SDE_forward(self, batch, stage=None, return_predicted=False):

    batch = Batch(batch, func_type="neural_SDE")
    X_hat = sdeint(self.func, batch.X0, **batch.t, **{"dt": 0.1})
    
    loss = self.loss_func(
        batch.X.contiguous(),
        X_hat.contiguous(),
        batch.W.contiguous(),
        batch.W.contiguous(),
    )
    
    if not stage == "predict":
        self.log("{}_loss".format(stage), loss["positional"].sum().item())

    if return_predicted:
        return {
            "loss": loss["positional"].sum(),
            "pred": X_hat,
            "batch": batch,
               }
    
    return {"loss": loss["positional"].sum()}
