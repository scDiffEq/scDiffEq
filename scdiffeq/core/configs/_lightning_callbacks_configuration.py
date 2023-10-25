
import os
import ABCParse
from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging

"""where built-in callbacks are configured"""

from .. import utils, callbacks


class InterTrainerEpochCounter(Callback):
    def __init__(self):
        ...
#         self.COMPLETED_EPOCHS = 0

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):

        pl_module.COMPLETED_EPOCHS += 1
        ce = pl_module.COMPLETED_EPOCHS

class LightningCallbacksConfiguration(ABCParse.ABCParse):
    def __init__(self):
        super().__init__()
        
        self.cbs = []

    @property
    def BuiltInCallbacks(self):

        return [
        ModelCheckpoint(
            every_n_epochs=self._every_n_epochs,
            save_on_train_epoch_end=True,
            save_top_k=self._save_top_k,
            save_last=self._save_last,
            monitor=self._monitor,
        ),
            InterTrainerEpochCounter(),
            callbacks.ModelLogging(),
#             callbacks.VisualizeTrackedLoss(
#                 **utils.extract_func_kwargs(func = callbacks.VisualizeTrackedLoss, kwargs = self._PARAMS),
#             ),
            # StochasticWeightAveraging(swa_lrs=self.swa_lrs),
            # considering rm SWA pending better understanding
        ]

    @property
    def Callbacks(self):
        return self.cbs + self.BuiltInCallbacks
        
    @property
    def GradientRetainedCallbacks(self):
        return [callbacks.GradientPotentialTest()] + self.cbs

    def __call__(
        self,
        version,
        viz_frequency = 1,
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
        
        [self.cbs.append(cb) for cb in callbacks]
        
        if retain_test_gradients:
            return self.GradientRetainedCallbacks
        return self.Callbacks
