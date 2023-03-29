
# -- import packages: ----------------------------------------------------------
from pytorch_lightning import LightningModule
from abc import ABC, abstractmethod
from autodevice import AutoDevice
import torch

NoneType = type(None)


# -- import local dependencies: ------------------------------------------------
from ...utils import AutoParseBase


# -- Parent Class: -------------------------------------------------------------
class BaseLightningDiffEq(LightningModule, AutoParseBase):
    """Pytorch-Lightning model trained within scDiffEq"""
    def __init__(self):
        super(BaseLightningDiffEq, self).__init__()
            
    @abstractmethod
    def process_batch(self):
        ...
        
    @abstractmethod
    def forward(self):
        ...
        
    @abstractmethod
    def integrate(self):
        ...
        
    @abstractmethod
    def loss(self):
        ...
        
    def record(self, loss, stage):
        """Record loss. called in step"""
        
        log_msg = "{}"
        if not isinstance(stage, NoneType):
            log_msg = f"{stage}_" + "{}"
        for i, l in enumerate(loss):
            self.log(log_msg.format(i), l.item())
            
    @abstractmethod
    def step(self, batch, batch_idx, stage=None):
        ...
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="training")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="validation")
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="test")
    
    def predict_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="predict")
    
    def configure_optimizers(self):
        return self.optimizers, self.lr_schedulers
