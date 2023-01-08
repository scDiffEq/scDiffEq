
from ..utils import Base


# -- LossLog: --------------------------------------------------------------------------
class LossLog(Base):

    def __init__(self, unlogged_stages=["predict", "test"], positional_loss_key="positional"):

        self.__parse__(locals())

    def positional(self, model, loss, stage, batch):
        
        if not stage in self.unlogged_stages:
        
            if not model.real_time:
                for i, t in enumerate(batch.t[:1]):
                    msg = "{}_loss_{}".format(stage, str(int(i)))
                    val = loss[self.positional_loss_key][i].item()
                    model.log(msg, val)
            else:
                for i, t in enumerate(batch.t):
                    msg = "{}_loss_{}".format(stage, str(int(i)))
                    val = loss[self.positional_loss_key][i].item()
                    model.log(msg, val)
