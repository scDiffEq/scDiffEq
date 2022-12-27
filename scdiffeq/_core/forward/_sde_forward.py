
# -- import packages: ------------------------------------------------------------------
from torchsde import sdeint


# -- import local dependencies: --------------------------------------------------------
from ._batch import Batch


# -- LossLog: --------------------------------------------------------------------------
class LossLog:
    def __parse__(self, kwargs, ignore=["self"]):

        for key, val in kwargs.items():
            if not key in ignore:
                setattr(self, key, val)

    def __init__(self, unlogged_stages=["predict", "test"], time_key="ts", positional_loss_key="positional"):

        self.__parse__(locals())

    def positional(self, model, loss, stage, batch):

        if not stage in self.unlogged_stages:
            for i, t in enumerate(batch.t[self.time_key]):
                msg = "{}_loss_{}".format(stage, str(int(t)))
                val = loss[self.positional_loss_key][i].item()
                model.log(msg, val)


loss_log = LossLog()

# -- default SDE forward: --------------------------------------------------------------
def SDE_forward(self, batch, stage=None, t=None):

    batch = Batch(batch, stage, func_type="NeuralSDE", t=t)
    X_hat = sdeint(self.func, batch.X0, **batch.t, **{"dt": 0.1})

    if not stage in ['predict']:
        loss = self.loss_func(
            batch.X.contiguous(),
            X_hat.contiguous(),
            batch.W.contiguous(),
            batch.W.contiguous(),
        )

        loss_log.positional(self, loss, stage, batch)

        if not stage in ["predict", "test"]:
            for i, t in enumerate(batch.t["ts"]):
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
