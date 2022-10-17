
__module_name__ = "_base_model.py"
__doc__ = """ To-Do """
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages ------------------------------------------------------------------------
from pytorch_lightning import LightningModule, loggers
import torch


# import local dependencies --------------------------------------------------------------
from . import _base_ancilliary as base
from . import _lightning_callbacks as cbs
from ._lightning_base import LightningBase


# BaseModel: -----------------------------------------------------------------------------
class BaseModel(LightningBase):
    def __init__(self, func, **kwargs):
        super(BaseModel, self).__init__(func, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.forward(self, batch, stage="train", dt=self.dt)

    def validation_step(self, batch, batch_idx):
        return self.forward(self, batch, stage="val", dt=self.dt)

    def test_step(self, batch, batch_idx):
        return self.forward(self, batch, stage="test", dt=self.dt)

    def predict_step(self, batch, batch_idx):
        return self.forward(self, batch, stage="predict", dt=self.dt)