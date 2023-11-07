
# -- import packages: ----------------------------------------------------------
from lightning import Trainer
from lightning.pytorch import loggers
import torch
import os
import ABCParse


# -- import local dependencies: ------------------------------------------------
from ._lightning_callbacks_configuration import LightningCallbacksConfiguration
from .. import utils, callbacks


# -- define typing: ------------------------------------------------------------
from typing import Union, Dict, List, Optional


# -- Main class: ---------------------------------------------------------------
class LightningTrainerConfiguration(ABCParse.ABCParse):
    def __init__(self, save_dir: str = "scDiffEq_Model"):
        super().__init__()
        
        self.__parse__(locals())
        if not os.path.exists(self._save_dir):
            os.mkdir(self._save_dir)

    # -- kwargs: ---------------------------------------------------------------
#     @property
#     def _CSVLogger_kwargs(self):
#         return ABCParse.function_kwargs(
#             func=loggers.CSVLogger, kwargs=self._PARAMS, ignore=['version'],
#         )

    @property
    def _Trainer_kwargs(self):
        ignore=["accelerator", "callbacks", 'version', 'logger']
        tk = ABCParse.function_kwargs(
            func=Trainer,
            kwargs=self._PARAMS,
        )
        for val in ignore:
            if val in tk.keys():
                tk.pop(val)
        return tk

    @property
    def Callbacks(self):
        callback_config = LightningCallbacksConfiguration()
        
        return callback_config(
            version = self._version,
            viz_frequency = self._viz_frequency,
            model_name=self._model_name,
            working_dir=self._working_dir,
            train_version=self._train_version,
            pretrain_version=self._pretrain_version,
            callbacks=self._callbacks,
            ckpt_frequency=self._ckpt_frequency,
            keep_ckpts=self._keep_ckpts,
            monitor=self._monitor,
            retain_test_gradients=self._retain_test_gradients,
            save_last=self._save_last_ckpt,
            # swa_lrs=1e-5,
        )

#         return callback_config(
#             callbacks=self._callbacks,
#             ckpt_frequency=self.ckpt_frequency,
#             keep_ckpts=self.keep_ckpts,
#             retain_test_gradients=self.retain_test_gradients,
#             monitor = self.monitor,
# #             swa_lrs = self.swa_lrs,
#             save_last = self.save_last_ckpt,
#             version = self.version,
#         )
    
    @property
    def accelerator(self):
        if not self._accelerator is None:
            return self._accelerator
        if torch.cuda.is_available():
            return "gpu"
        if torch.backends.mps.is_available(): # experimental
            return "mps"
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
            logger=self._logger, # loggers.CSVLogger(**self._CSVLogger_kwargs),
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
            logger=self._logger,# loggers.CSVLogger(**self._CSVLogger_kwargs),
            num_sanity_val_steps=-1,
            enable_progress_bar=False,
            **self._Trainer_kwargs,
        )

    def __call__(
        self,
        logger,
        lr: float = None,
        model_name="scDiffEq_model",
        working_dir=os.getcwd(),
        train_version=0,
        pretrain_version=0,
        viz_frequency = 1,
        stage=None,
        max_epochs=500,
        monitor=None,
        accelerator=None,
        devices=1,
        prefix: str = "",
        log_every_n_steps=1,
        flush_logs_every_n_steps: int = 1,
        ckpt_frequency: int = 25,
        save_last_ckpt: bool = True,
        keep_ckpts: int = -1,
        version: Optional[Union[int, str]] = None,
        callbacks: list = [],
        potential_model: bool = False,
        check_val_every_n_epoch = 1,
        limit_val_batches = None,
        num_sanity_val_steps = None,
        val_check_interval = None,
#         swa_lrs: float = None,
        reload_dataloaders_every_n_epochs = 1,
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
        
        self._logger = logger

        self._retain_test_gradients = False
        if stage is None:
            stage = ""
            
#         if isinstance(swa_lrs, NoneType):
#             swa_lrs = lr
            
        if torch.cuda.device_count() > 0:
            devices = torch.cuda.device_count()

        self.__parse__(locals()) #  private=['accelerator', 'callbacks'])
        
#         self._PARAMS["name"] = "{}_logs".format(stage)
#         log_save_dir = os.path.join(self.save_dir, self._PARAMS["name"])
#         if not os.path.exists(log_save_dir):
#             os.mkdir(log_save_dir)

        if (potential_model) and (stage in ["test", "predict"]):
            self._retain_test_gradients = True
            return self.GradientsRetainedTestTrainer

        return self.Trainer
