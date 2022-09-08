
__module_name__ = "_CustomModel.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import local dependencies ---------------------------------------------------
from ._core._BaseModel import BaseModel
from pytorch_lightning import Trainer


# MAIN MODULE CLASS: CustomModel ----------------------------------------------
class CustomModel(BaseModel):
    
    def __init__(self,
                 adata=None,
                 dataset=None,
                 func=None,
                 accelerator="gpu",
                 devices=1,
                 epochs=1500,
                 log_every_n_steps=1,
                ):
        """
        Parameters:
        -----------
        adata
        
        func
        
        accelerator
        
        devices
        
        epochs
        
        log_every_n_steps
        
        Returns:
        --------
        None
        
        Notes:
        ------
        """
        super(CustomModel, self).__init__()
        
        if adata:
            self._adata = adata
            # TO-DO: prepare torch.utils.data.dataset class from adata
            
        self.model_setup(dataset, func)
        self._accelerator = accelerator
        self._log_every_n_steps = log_every_n_steps
        self._devices = devices
        self.trainer = Trainer(
                          accelerator=self._accelerator,
                          devices=self._devices,
                          max_epochs=epochs,
                          logger=self._logger,
                          log_every_n_steps=log_every_n_steps,
                          callbacks=self._callback_list,
                         )
        
    def fit(self):
        
        
        self.trainer.fit(self, self.dataset)
        
        
    def evaluate(self):
        
        ...
        
# ------------------------------------------------------------------------------------- #