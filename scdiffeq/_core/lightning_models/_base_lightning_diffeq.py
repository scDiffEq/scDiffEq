
# -- import packages: ----------------------------------------------------------
from pytorch_lightning import LightningModule
from abc import ABC, abstractmethod


# -- Model Base Class: ---------------------------------------------------------
class BaseLightningDiffEq(ABC, LightningModule):
    """Pytorch-Lightning model trained within scDiffEq"""
    def __init__(self):
        super(BaseLightningDiffEq, self).__init__()
        ...
    
    @abstractmethod
    def process_batch(self):
        ...
        
    @abstractmethod
    def forward(self):
        ...
        
    @abstractmethod
    def loss(self):
        ...
        
    @abstractmethod
    def record(self):
        ...    

    @abstractmethod
    def step(self, batch, batch_idx, stage=None):
        ...
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="training")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, stage="validation")
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, stage="test")
    
    def predict_step(self, batch, batch_idx):
        return self.step(batch, stage="predict")
    
    def configure_optimizers(self):
        return self.optimizers, self.lr_schedulers
