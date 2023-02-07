
# -- import packages: ----------------------------------------------------------
from pytorch_lightning import Trainer, loggers
import torch
import os


# -- import local dependencies: ------------------------------------------------
from ._lightning_callbacks_configuration import LightningCallbacksConfiguration
from .. import utils, callbacks


# -- define typing: ------------------------------------------------------------
from typing import Union, Dict, List
NoneType = type(None)


# -- Main class: ---------------------------------------------------------------
class LightningTrainerConfiguration(utils.AutoParseBase):

    """
    Container class to instantiate trainers as needed.

    Logger and Callbacks are also configured here since they are passed through the trainer.
    """

    def __init__(
        self,
        save_dir: str = "scDiffEq_Model",
    ):
        self.__parse__(locals())

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
            retain_test_gradients=self.retain_test_gradients,
        )
    
    @property
    def accelerator(self):
        if not isinstance(self._accelerator, NoneType):
            return self._accelerator
        if torch.cuda.is_available():
            return "gpu"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # -- trainers: -------------------------------------------------------------
    @property
    def Trainer(self):
        """
        Main Lightning Trainer used for fitting / testing.
        If pre-train routine was used, Trainer loads from ckpt path.
        """

#         self._Trainer_kwargs["callbacks"] = self.Callbacks
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
        stage=None,
        max_epochs=500,
        accelerator=None,
        devices=torch.cuda.device_count(),
        prefix: str = "",
        log_every_n_steps=1,
        flush_logs_every_n_steps: int = 1,
        version: Union[int, str, NoneType] = None,
        callbacks: list = [],
        potential_model: bool = False,
        **kwargs
    ):

        self.retain_test_gradients = False
        if isinstance(stage, NoneType):
            stage = ""

        self.__parse__(locals(), private=['accelerator', 'callbacks'])
        self._KWARGS["name"] = "{}_logs".format(stage)
        log_save_dir = os.path.join(self.save_dir, self._KWARGS["name"])
        if not os.path.exists(log_save_dir):
            os.mkdir(log_save_dir)

        if (potential_model) and (stage in ["test", "predict"]):
            self.retain_test_gradients = True
            return self.GradientsRetainedTestTrainer

        return self.Trainer
