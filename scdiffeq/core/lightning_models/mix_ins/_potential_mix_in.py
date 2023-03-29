
import torch

class PotentialMixIn(object):
    
    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        return self.step(batch, batch_idx, stage="validation")
    
    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        return self.step(batch, batch_idx, stage="test")
    
    def predict_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        return self.step(batch, batch_idx, stage="predict")