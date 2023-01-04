

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


# from ._loss_log import LossLog
# loss_log = LossLog()