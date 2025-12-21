# -- import packages: ---------------------------------------------------------
import lightning
import logging

# -- import local dependencies: -----------------------------------------------
from .. import utils

# -- set type hints: ----------------------------------------------------------
from typing import List, Optional

# -- configure logging: --------------------------------------------------------
logger = logging.getLogger(__name__)


# -- base mix-in cls: ---------------------------------------------------------
class BaseRoutineMixIn(object):
    """Base train/pre-train routine container"""

    @property
    def _TRAINING_FUNCS(self):
        """"""
        return [self.TrainerGenerator, lightning.Trainer]

    def _check_disable_validation(self, trainer_kwargs) -> None:
        """"""
        if self._train_val_split[1] == 0:
            trainer_kwargs.update(
                {
                    "check_val_every_n_epoch": 0,
                    "limit_val_batches": 0.0,
                    "num_sanity_val_steps": 0.0,
                    "val_check_interval": 0.0,
                },
            )

    def _update_trainer_kwargs(
        self, kwargs, ignore=["version", "working_dir", "logger"]
    ) -> None:
        # -- update kwargs with params already passed (this is problematic)
        kwargs.update(self._PARAMS)

        self._TRAINER_KWARGS = {}
        for func in self._TRAINING_FUNCS:
            self._TRAINER_KWARGS.update(
                utils.extract_func_kwargs(
                    func=func,
                    kwargs=kwargs,
                    ignore=ignore,
                )
            )

        self._check_disable_validation(self._TRAINER_KWARGS)

    def _flag_log_dir(self) -> None:
        if hasattr(self, "_csv_logger"):
            logger.debug(f"Logging locally to: {self._csv_logger.log_dir}")

    def _initialize__attrs(self):
        """ """
        for attr in ["PRETRAIN", "TRAIN"]:
            attr_name = f"_{attr}_CONFIG_COUNT"
            if not hasattr(self, attr_name):
                setattr(self, attr_name, 0)


# -- pre-training mix-in: -----------------------------------------------------
class PreTrainMixIn(BaseRoutineMixIn):
    """Container hosting the pre-training config/execution methods"""

    def _configure_pretrain_step(self, kwargs) -> None:
        """"""

        logger.debug("Configuring pretraining step")

        self._update_trainer_kwargs(kwargs)

        STAGE = "pretrain"

        self._initialize__attrs()

        # -- generate a new trainer
        self.pretrainer = self.TrainerGenerator(
            logger=self.logger,
            max_epochs=self._pretrain_epochs,
            stage=STAGE,
            working_dir=self._working_dir,
            version=self.version,
            pretrain_version=self._PRETRAIN_CONFIG_COUNT,
            train_version=self._TRAIN_CONFIG_COUNT,
            callbacks=self._pretrain_callbacks,
            monitor_hardware=self._monitor_hardware,
            **self._TRAINER_KWARGS,
        )

        self.__flag_log_dir()

        self._PRETRAIN_CONFIG_COUNT += 1

    def pretrain(
        self,
        pretrain_epochs: int = 500,
        pretrain_callbacks: Optional[List] = [],
        ckpt_frequency: int = 25,
        save_last_ckpt: bool = True,
        keep_ckpts: int = -1,
        monitor: Optional[str] = None,
        accelerator: Optional[str] = None,
        log_every_n_steps: int = 1,
        reload_dataloaders_every_n_epochs: int = 1,
        devices: Optional[int] = None,
        deterministic: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """Pretrain method.

        Extended description of the pretrain method.

        Args:
            pretrain_epochs (int): Description. **Default**: 500

            pretrain_callbacks (Optional[List]): Description. **Default**: []

            ckpt_frequency (int): Description. **Default**: 25
        """

        self.__update__(locals())

        if self._pretrain_epochs > 0:
            self._configure_pretrain_step(locals())
            self.pretrainer.fit(self.DiffEq, self.LitDataModule)


# -- training mix-in: ---------------------------------------------------------
class TrainMixIn(BaseRoutineMixIn):
    """Container hosting the training config/execution methods"""

    def _configure_train_step(self, kwargs):
        """"""
        logger.debug(f"Configuring training step")

        self._update_trainer_kwargs(kwargs)

        STAGE = "train"

        self._initialize__attrs()

        # -- generate a new trainer
        self.trainer = self.TrainerGenerator(
            logger=self.logger,
            max_epochs=self._train_epochs,
            stage=STAGE,
            working_dir=self._working_dir,
            version=self.version,
            pretrain_version=self._PRETRAIN_CONFIG_COUNT,
            train_version=self._TRAIN_CONFIG_COUNT,
            callbacks=self._train_callbacks,
            **self._TRAINER_KWARGS,
        )

        self._flag_log_dir()

        self._TRAIN_CONFIG_COUNT += 1

    def train(
        self,
        train_epochs: int = 500,
        train_callbacks: Optional[List] = [],
        ckpt_frequency: int = 25,
        save_last_ckpt: bool = True,
        keep_ckpts: int = -1,
        monitor: Optional[str] = None,
        accelerator: Optional[str] = None,
        log_every_n_steps: int = 1,
        reload_dataloaders_every_n_epochs: int = 1,
        devices: Optional[int] = 1,
        deterministic: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """ """

        self.__update__(locals())

        if self._train_epochs > 0:
            self._configure_train_step(locals())
            self.trainer.fit(self.DiffEq, self.LitDataModule)
