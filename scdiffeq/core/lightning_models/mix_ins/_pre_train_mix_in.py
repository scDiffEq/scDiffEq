
from abc import abstractmethod

class PreTrainMixIn(object):
    """
    Requires user to:
    1. pass pretrain_epochs to the model __init__
    2. def `pretrain_step` method on the model
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx, *args, **kwargs):
        if self.PRETRAIN:
            return self.pretrain_step(batch, batch_idx, stage="pretrain")
        return self.step(batch, batch_idx, stage="training")

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        if not self.PRETRAIN:
            return self.step(batch, batch_idx, stage="validation")

    @property
    def PRETRAIN(self):
        return self.COMPLETED_EPOCHS < self.hparams["pretrain_epochs"]

    @abstractmethod
    def pretrain_step(self):
        ...