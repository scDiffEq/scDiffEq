# -- import packages: ----------------------------------------------------------
import ABCParse
import lightning
import logging
import os
import torch

# -- import local dependencies: ------------------------------------------------
from ._lightning_callbacks_configuration import LightningCallbacksConfiguration
from ._progress_bar_config import ProgressBarConfig

# -- set type hints: ----------------------------------------------------------
from typing import List, Optional, Union

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)

# -- constants: ---------------------------------------------------------------
INTERACTIVE_ENVS = ["jupyter", "colab"]


# -- supporting functions: ----------------------------------------------------
def _n_requested_devices(devices: Union[int, str, List[int]]) -> int:
    """Number of devices asked for by a Lightning ``devices`` spec."""
    if isinstance(devices, int):
        return devices
    if isinstance(devices, (list, tuple)):
        return len(devices)
    if isinstance(devices, str):
        return len([device for device in devices.split(",") if device.strip()])
    return 1


# -- Main class: --------------------------------------------------------------
class LightningTrainerConfiguration(ABCParse.ABCParse):
    def __init__(self, save_dir: str = "scDiffEq_Model", print_every: int = 10):
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
        ignore = ["accelerator", "callbacks", "version", "logger"]
        tk = ABCParse.function_kwargs(
            func=lightning.pytorch.Trainer,
            kwargs=self._PARAMS,
        )
        for val in ignore:
            if val in tk.keys():
                tk.pop(val)
        return tk

    @property
    def Callbacks(self):
        callback_config = LightningCallbacksConfiguration()

        callbacks = callback_config(
            version=self._version,
            monitor_hardware=self._monitor_hardware,
            stage=self._stage,
            viz_frequency=self._viz_frequency,
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
        callbacks.extend(self._progress_bar_config.pbar)
        return callbacks

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
    def _is_interactive(self) -> bool:
        """True when running inside a notebook kernel (Jupyter or Colab)."""
        return self._progress_bar_config.env in INTERACTIVE_ENVS

    def _resolve_devices(
        self, devices: Optional[Union[int, str, List[int]]]
    ) -> Union[int, str, List[int]]:
        """Resolve the ``devices`` spec handed to the Lightning Trainer.

        ``None``, ``"auto"`` and ``-1`` mean every visible CUDA device, or one
        device when there are none. Any other value is honored as given.

        Multi-device training is capped to one device inside a notebook:
        Lightning launches it by forking the kernel, and a forked process
        cannot re-initialize CUDA (#111).
        """
        n_cuda = torch.cuda.device_count()

        if devices is None or devices in ["auto", -1, "-1"]:
            devices = n_cuda if n_cuda > 0 else 1

        n_requested = _n_requested_devices(devices)

        if n_requested > 1:
            if self._is_interactive:
                logger.warning(
                    f"{n_requested} devices requested from a notebook: using 1. "
                    "Lightning trains on multiple devices by forking the kernel, "
                    "and CUDA cannot be re-initialized in a forked process. Run "
                    "as a script to train on multiple devices."
                )
                return 1
            logger.info(f"Training on {n_requested} devices.")

        return devices

    @property
    def accelerator(self):
        if not self._accelerator is None:
            return self._accelerator
        if torch.cuda.is_available():
            return "gpu"
        if torch.backends.mps.is_available():  # experimental
            return "mps"
        return "cpu"

    # -- trainers: -------------------------------------------------------------
    @property
    def Trainer(self):
        """
        Main Lightning Trainer used for fitting / testing.
        If pre-train routine was used, Trainer loads from ckpt path.
        """

        return lightning.pytorch.Trainer(
            accelerator=self.accelerator,
            logger=self._logger,  # loggers.CSVLogger(**self._CSVLogger_kwargs),
            callbacks=self.Callbacks,
            enable_progress_bar=self._progress_bar_config.enable_progress_bar,
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

        return lightning.pytorch.Trainer(
            accelerator=self.accelerator,
            logger=self._logger,  # loggers.CSVLogger(**self._CSVLogger_kwargs),
            num_sanity_val_steps=-1,
            enable_progress_bar=self._progress_bar_config.enable_progress_bar,
            **self._Trainer_kwargs,
        )

    def __call__(
        self,
        logger,
        lr: float = None,
        model_name="scDiffEq_model",
        gradient_clip_val: float = 0.5,
        monitor_hardware: bool = False,
        working_dir=os.getcwd(),
        train_version=0,
        pretrain_version=0,
        viz_frequency=1,
        stage=None,
        max_epochs=500,
        print_every=10,
        monitor=None,
        accelerator=None,
        devices: Optional[Union[int, str, List[int]]] = None,
        prefix: str = "",
        log_every_n_steps=1,
        flush_logs_every_n_steps: int = 1,
        ckpt_frequency: int = 25,
        save_last_ckpt: bool = True,
        keep_ckpts: int = -1,
        version: Optional[Union[int, str]] = None,
        callbacks: list = [],
        potential_model: bool = False,
        check_val_every_n_epoch=1,
        limit_val_batches=None,
        num_sanity_val_steps=None,
        val_check_interval=None,
        #         swa_lrs: float = None,
        reload_dataloaders_every_n_epochs=1,
        **kwargs,
    ):
        """

        Args:
        stage (str): stage of the training.
        max_epochs (int): maximum number of epochs to train.
        callbacks (list): list of callbacks to use.
        potential_model (bool): whether the model is a potential model.
        check_val_every_n_epoch (int): number of epochs to check validation.
        limit_val_batches (int): number of validation batches to use.
        num_sanity_val_steps (int): number of sanity validation steps.
        val_check_interval (int): number of epochs to check validation.
        reload_dataloaders_every_n_epochs (int): number of epochs to reload dataloaders.
        **kwargs: additional keyword arguments to pass to the Trainer.

        Returns: TrainerGenerator
        """

        self._gradient_clip_val = gradient_clip_val
        self._logger = logger

        self._retain_test_gradients = False
        if stage is None:
            stage = ""
        self._stage = stage
        self._progress_bar_config = ProgressBarConfig(total_epochs=max_epochs, print_every=print_every)

        devices = self._resolve_devices(devices)

        self.__parse__(locals())

        if (potential_model) and (stage in ["test", "predict"]):
            self._retain_test_gradients = True
            return self.GradientsRetainedTestTrainer

        return self.Trainer
