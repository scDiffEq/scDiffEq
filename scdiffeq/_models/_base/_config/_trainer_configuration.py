
# -- import packages: --------------------------------------------------------------------
import pytorch_lightning
import torch
import os


# -- import local dependencies: ----------------------------------------------------------
from .._utils import parser


# -- Main module class: ------------------------------------------------------------------
class TrainerConfiguration:
    def __init__(
        self,
        model_save_dir: str = "scDiffEq_model",
        log_name: str = "lightning_logs",
        version=None,
        prefix="",
        flush_logs_every_n_steps=5,
        max_epochs=1500,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=5,
        **kwargs
    ):
        """
        
        Parameters:
        -----------
        model_save_dir
        
        log_name
        
        version
        
        prefix
        
        flush_logs_every_n_steps
        
        max_epochs
        
        log_every_n_steps
        
        reload_dataloaders_every_n_epochs
        
        **kwargs: any additional kwarg accepted by pytorch_lightning.Trainer

        Notes:
        ------
        (1) Used as the default logger because it is the least complex and most predictable.
        (2) This function simply handle the args to pytorch_lighting.loggers.CSVLogger. While
            not functionally necessary, helps to clean up model code a bit.
        (3) doesn't change file names rather the logger name and what's added to the front of
            each log name event within the model.
        (4) Versioning contained / created automatically within by lightning logger
        """
        
        self.kwargs, self.trainer_kwargs = parser(self, locals(), func=pytorch_lightning.Trainer)        
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

    @property
    def accelerator(self):
        if torch.cuda.is_available():
            return "gpu"
        return "cpu"

    @property
    def devices(self):
        return torch.cuda.device_count()

    @property
    def CSVLogger(self):     
        return pytorch_lightning.loggers.CSVLogger(
            save_dir=self.model_save_dir,
            name=self.log_name,
            version=self.version,
            prefix=self.prefix,
            flush_logs_every_n_steps=self.flush_logs_every_n_steps,
        )
    
    @property
    def log_path(self):
        return os.path.join(self.model_save_dir, self.log_name)
    
    
    @property
    def trainer(self):
        return pytorch_lightning.Trainer(
            accelerator=self.accelerator, logger=self.CSVLogger, **self.trainer_kwargs
        )