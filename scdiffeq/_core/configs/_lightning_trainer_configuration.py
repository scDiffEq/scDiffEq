
__module_name__ = "_lightning_trainer_configuration.py"
__doc__ = """To-do"""
__author__ = "Michael E. Vinyard"
__email__ = "mvinyard@broadinstitute.org"


# -- import packages: --------------------------------------------------------------------
import pytorch_lightning
import torch
import os


# -- import local dependencies: ----------------------------------------------------------
from ..utils import function_kwargs
from .. import lightning_callbacks as callbacks


# -- Main class: -------------------------------------------------------------------------
class LightningTrainerConfig:
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
        save_fitting_loss_img=True,
        ckpt_outputs_frequency=50,
        **kwargs
    ):
        """
        Configure Trainer, including the logger (CSVLogger).

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
        self.__configure__(locals())

    # -- methods: ------------------------------------------------------------------------
    def __parse__(self, kwargs, ignore=["self"]):
        for key, val in kwargs.items():
            if not key in ignore:
                if key == "kwargs":
                    for _key, _val in val.items():
                        setattr(self, _key, _val)
                else:
                    setattr(self, key, val)

        self._trainer_kwargs = function_kwargs(
            func=pytorch_lightning.Trainer, kwargs=kwargs
        )

    def _setup_model_save_dir(self):
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

    def __configure__(self, kwargs, ignore=["self"]):

        self.__parse__(kwargs, ignore)
        self._setup_model_save_dir()

    def _accelerator(self):
        if torch.cuda.is_available():
            return "gpu"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    # -- properties: ---------------------------------------------------------------------
    @property
    def custom_callbacks(self):
        return [callbacks.LossAccounting(), callbacks.IntermittentSaves(self.ckpt_outputs_frequency)]
        
    @property
    def callbacks(self):
        if not hasattr(self, "_callbacks"):
            self._callbacks = []
        for cb in self.custom_callbacks:
            self._callbacks.append(cb)
        return self._callbacks
    
    @property
    def trainer_kwargs(self):
        return self._trainer_kwargs

    @property
    def log_path(self):
        return os.path.join(self._model_save_dir, self._log_name)

    @property
    def accelerator(self):
        return self._accelerator()

    @property
    def n_devices(self):
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
    def trainer(self):
        return pytorch_lightning.Trainer(
            accelerator=self.accelerator, logger=self.CSVLogger, callbacks=self.callbacks, **self.trainer_kwargs
        )
