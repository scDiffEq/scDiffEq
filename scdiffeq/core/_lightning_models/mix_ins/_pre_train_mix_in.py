
# -- import packages: ---------------------------------------------
from abc import abstractmethod

# -- import local dependencies: ----
from ._base_mix_in import BaseMixIn


class PreTrainMixIn(BaseMixIn):
    """Overrides the training_step from BaseLightningSDE to add a diversion towards the pre-train routine."""

    def __init__(self, *args, **kwargs):
        super(PreTrainMixIn, self).__init__()

    def training_step(self, batch, batch_idx):
        
        """Gives the option to do a separate routine designated as `pretrain_step`"""

        if self.current_epoch < self.pretrain_epochs:
            return self.pretrain_step(batch, batch_idx, stage="pretraining_train")

        return self.step(batch, batch_idx, stage="training")

    def validation_step(self, batch, batch_idx):
        
        """Gives the option to do a separate routine designated as `pretrain_step`"""

        if self.current_epoch < self.pretrain_epochs:
            return self.pretrain_step(batch, batch_idx, stage="pretraining_validation")

        return self.step(batch, batch_idx, stage="validation")

    @abstractmethod
    def pretrain_step(self, batch, batch_idx, stage="pretraining"):
        """Called within {training/validation}_step Should return loss.sum()"""
        ...
