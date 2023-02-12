
import torch

from ._base_lightning_diffeqs import BaseLightningDiffEq

class PotentialMixIn(BaseLightningDiffEq):
    
    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        return self.step(batch, batch_idx, stage="validation")
    
    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        return self.step(batch, batch_idx, stage="test")
    
    def predict_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        return self.step(batch, batch_idx, stage="predict")