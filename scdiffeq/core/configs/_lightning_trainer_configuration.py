
# -- import packages: ----------------------------------------------------------
# from pytorch_lightning import Trainer, loggers
from lightning import Trainer
from lightning.pytorch import loggers
import torch
import os


# from licorice_font import font_format

# debug_fmt = font_format("DEBUG", ['YELLOW'])
# debug_msg = f"- {debug_fmt} | "


# -- import local dependencies: ------------------------------------------------
from ._lightning_callbacks_configuration import LightningCallbacksConfiguration
from .. import utils, callbacks


# -- define typing: ------------------------------------------------------------
from typing import Union, Dict, List
NoneType = type(None)


# -- Main class: ---------------------------------------------------------------
class LightningTrainerConfiguration(utils.AutoParseBase):
    def __init__(
        self,
        save_dir: str = "scDiffEq_Model",
    ):
        self.__parse__(locals())
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # -- kwargs: ---------------------------------------------------------------
    @property
    def _CSVLogger_kwargs(self):
        return utils.extract_func_kwargs(func=loggers.CSVLogger, kwargs=self._KWARGS)

    @property
    def _Trainer_kwargs(self):
        return utils.extract_func_kwargs(func=Trainer, kwargs=self._KWARGS)

    @property
    def Callbacks(self):
        callback_config = LightningCallbacksConfiguration()
        return callback_config(
            callbacks=self._callbacks,
            ckpt_frequency=self.ckpt_frequency,
            keep_ckpts=self.keep_ckpts,
            retain_test_gradients=self.retain_test_gradients,
            monitor = self.monitor,
#             swa_lrs = self.swa_lrs,
            save_last = self.save_last_ckpt,
        )
    
    @property
    def accelerator(self):
        if not isinstance(self._accelerator, NoneType):
            return self._accelerator
        if torch.cuda.is_available():
            return "gpu"
        # would love to include mps however, currently (torch v1.13)
        # does not have enough compatability with libraries we depend
        # on to be used with Apple Silicon.
        # if torch.backends.mps.is_available():
        #     return "mps"
        return "cpu"

    # -- trainers: -------------------------------------------------------------
    @property
    def Trainer(self):
        """
        Main Lightning Trainer used for fitting / testing.
        If pre-train routine was used, Trainer loads from ckpt path.
        """
        return Trainer(
            accelerator=self.accelerator,
            logger=loggers.CSVLogger(**self._CSVLogger_kwargs),
            callbacks=self.Callbacks,
            **self._Trainer_kwargs,
        )

    @property
    def GradientsRetainedTestTrainer(self):
        """
        Quasi test trainer - serves as a workaround for evaluating test data
        while retaining gradients.
        """

        self._Trainer_kwargs["max_epochs"] = 0
        self._Trainer_kwargs["callbacks"] = self.Callbacks

        return Trainer(
            accelerator=self.accelerator,
            logger=loggers.CSVLogger(**self._CSVLogger_kwargs),
            num_sanity_val_steps=-1,
            enable_progress_bar=False,
            **self._Trainer_kwargs,
        )

    def __call__(
        self,
        lr: float = None,
        stage=None,
        max_epochs=500,
        monitor=None,
        accelerator=None,
        devices=None,
        prefix: str = "",
        log_every_n_steps=1,
        flush_logs_every_n_steps: int = 1,
        ckpt_frequency: int = 25,
        save_last_ckpt: bool = True,
        keep_ckpts: int = -1,
        version: Union[int, str, NoneType] = None,
        callbacks: list = [],
        potential_model: bool = False,
#         swa_lrs: float = None,
        **kwargs
    ):
        
        """
        Return trainer upon call.
        
        Parameters:
        -----------
        stage
            type: str
            default: None
        
        max_epochs
            type: int
            default: 500
        
        accelerator
            type: str
            default: None
        
        devices
            type: int
            default: None
        
        prefix
            type: str
            default: ""
        
        log_every_n_steps
            type: int
            default: 1
        
        flush_logs_every_n_steps
            type: int
            default: 1
        
        version
            type: [int, str]
            default: None
        
        callbacks
            type: list
            default: []
        
        potential_model
            type: bool
            default: False
            
        kwargs:
        -------
        Keyword arguments that may be passed to pytorch_lightning.Trainer()
        
        Notes:
        ------
        """

        self.retain_test_gradients = False
        if isinstance(stage, NoneType):
            stage = ""
            
#         if isinstance(swa_lrs, NoneType):
#             swa_lrs = lr
            
        if torch.cuda.device_count() > 0:
            devices = torch.cuda.device_count()

        self.__parse__(locals(), private=['accelerator', 'callbacks'])
        self._KWARGS["name"] = "{}_logs".format(stage)
        log_save_dir = os.path.join(self.save_dir, self._KWARGS["name"])
        if not os.path.exists(log_save_dir):
            os.mkdir(log_save_dir)

        if (potential_model) and (stage in ["test", "predict"]):
            self.retain_test_gradients = True
            return self.GradientsRetainedTestTrainer

        return self.Trainer
