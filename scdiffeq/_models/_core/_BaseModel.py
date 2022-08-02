
__module_name__ = "_BaseModel.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
from pytorch_lightning import LightningModule


class BaseModel(LightningModule):
    
    def __init__(self, lr=1e-3):
        
        self._lr = lr
        
    def forward(self):
        
        """TO-DO"""
        
    def training_step(self):
        
        """
        TO-DO
        
        Notes:
        ------
        (1) Required method of the pytorch_lightning.LightningModule subclass.
        """
        
    def validation_step(self):
        
        """TO-DO"""
        
    def test_step(self):
        
        """TO-DO"""
        
        
    def configure_optimizers(self):
        """
        
        To-Do:
        ------
        (1) Multiple optimizer configuration
        (2) LR-scheduler (and multiple LR-scheduler configuration)
        
        Notes:
        ------
        (1) Required method of the pytorch_lightning.LightningModule subclass.
        """
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self._lr)