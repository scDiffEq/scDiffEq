
from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging


from .. import utils, callbacks


class LightningCallbacksConfiguration(utils.AutoParseBase):
    def __init__(self):
        self.cbs = []

    @property
    def BuiltInCallbacks(self):
        return [
        ModelCheckpoint(
            every_n_epochs=self.every_n_epochs,
            save_on_train_epoch_end=True,
            save_top_k=self.save_top_k,
            save_last=self.save_last,
            monitor=self.monitor,
        ),
#         StochasticWeightAveraging(swa_lrs=self.swa_lrs), # considering removal pending better understanding
        ]

    @property
    def Callbacks(self):
        return self.cbs + self.BuiltInCallbacks

    @property
    def GradientRetainedCallbacks(self):
        return [callbacks.GradientPotentialTest()] + self.cbs

    def __call__(
        self,
        callbacks=[],
        ckpt_frequency=10,
        keep_ckpts=-1,
        swa_lrs=1e-5,
        monitor=None,
        retain_test_gradients=False,
        save_last=True,
    ):
        
        self.every_n_epochs = ckpt_frequency
        self.save_top_k = keep_ckpts
        self.save_last = save_last
        self.monitor = monitor
        self.swa_lrs = swa_lrs

        [self.cbs.append(cb) for cb in callbacks]
        if retain_test_gradients:
            return self.GradientRetainedCallbacks
        return self.Callbacks
