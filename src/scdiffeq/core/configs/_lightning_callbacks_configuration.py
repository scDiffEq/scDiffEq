# -- import packages: ---------------------------------------------------------
import ABCParse
import lightning
import os

# -- import local dependencies: -----------------------------------------------
from .. import utils, callbacks as _callbacks

# -- set type hints: ----------------------------------------------------------
from typing import List


# -- supporting function: -----------------------------------------------------
def in_jupyter_notebook():
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except (NameError, ImportError):
        return False  # Probably standard Python interpreter


# -- supporting class: --------------------------------------------------------
class InterTrainerEpochCounter(lightning.pytorch.callbacks.Callback):
    def __init__(self): ...

    #         self.COMPLETED_EPOCHS = 0

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):

        pl_module.COMPLETED_EPOCHS += 1
        ce = pl_module.COMPLETED_EPOCHS


# -- operational class: -------------------------------------------------------
class LightningCallbacksConfiguration(ABCParse.ABCParse):
    def __init__(self):
        super().__init__()

        self.cbs = []

    @property
    def BuiltInCallbacks(self):

        return [
            lightning.pytorch.callbacks.ModelCheckpoint(
                every_n_epochs=self._every_n_epochs,
                save_on_train_epoch_end=True,
                save_top_k=self._save_top_k,
                save_last=self._save_last,
                monitor=self._monitor,
            ),
            InterTrainerEpochCounter(),
            _callbacks.ModelLogging(),
        ]

    @property
    def BuiltInPreTrainCallbacks(self):

        return [
            lightning.pytorch.callbacks.ModelCheckpoint(
                every_n_epochs=self._every_n_epochs,
                save_on_train_epoch_end=True,
                save_top_k=self._save_top_k,
                save_last=self._save_last,
                monitor=self._monitor,
            ),
            InterTrainerEpochCounter(),
        ]

    @property
    def DEPLOYED_CALLBACKS(self):
        return self.cbs + self.BuiltInCallbacks

    def _update_callbacks(
        self, callbacks: List[lightning.pytorch.callbacks.Callback]
    ) -> None:

        [self.cbs.append(cb) for cb in callbacks]

        if self._monitor_hardware:
            self.cbs.append(_callbacks.MemoryMonitor())
        if self._retain_test_gradients:
            self.cbs.append(_callbacks.GradientPotentialTest())

    def __call__(
        self,
        version,
        stage,
        monitor_hardware: bool = False,
        viz_frequency=1,
        model_name="scDiffEq_model",
        working_dir=os.getcwd(),
        train_version=0,
        pretrain_version=0,
        callbacks=[],
        ckpt_frequency=10,
        keep_ckpts=-1,
        swa_lrs=1e-5,
        monitor=None,
        retain_test_gradients=False,
        save_last=True,
    ):

        self.__parse__(locals(), ignore=["callbacks"])

        self._every_n_epochs = ckpt_frequency
        self._save_top_k = keep_ckpts

        self._update_callbacks(callbacks=callbacks)

        return self.DEPLOYED_CALLBACKS
