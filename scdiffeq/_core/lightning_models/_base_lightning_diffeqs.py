
# -- import packages: ----------------------------------------------------------
from pytorch_lightning import LightningModule
from abc import ABC, abstractmethod
from brownian_diffuser import nn_int
from torchdiffeq import odeint
from torchsde import sdeint
import torch


# -- Parent Class: -------------------------------------------------------------
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


# -- base sub-classes: ---------------------------------------------------------
class BaseLightningDriftNet(BaseLightningDiffEq):
    def __init__(self, dt=0.1, stdev=torch.Tensor([0.5])):
        super(BaseLightningDriftNet, self).__init__()

        self.save_hyperparameters()

    def forward(self, X0, t, stage=None, max_steps=None, return_all=False):
        return nn_int(
            self.func,
            X0=X0,
            t=t,
            dt=self.hparams["dt"],
            stdev=self.hparams["stdev"],
            max_steps=max_steps,
            return_all=return_all,
        )

class BaseLightningODE(BaseLightningDiffEq):
    def __init__(self):
        super(BaseLightningODE, self).__init__()

    def forward(self, X0, t, stage=None, **kwargs):
        """
        We want this to be easily-accesible from the outside, so we
        directly define the forward step with the integrator code.
        """
        
        return odeint(self.func, X0, t=t, **kwargs)

class BaseLightningSDE(BaseLightningDiffEq):
    def __init__(self, dt=0.1):
        super(BaseLightningSDE, self).__init__()

    def forward(self, X0, t, stage=None, **kwargs):
        """
        We want this to be easily-accesible from the outside, so we
        directly define the forward step with the integrator code.
        """
        
        return sdeint(self.func, X0, ts=t, dt=self.hparams["dt"], **kwargs)
