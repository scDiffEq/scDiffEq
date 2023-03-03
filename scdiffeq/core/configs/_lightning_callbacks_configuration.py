
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging


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
            monitor=self.monitor,
        ),
        StochasticWeightAveraging(swa_lrs=self.swa_lrs),
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
    ):
        
        self.every_n_epochs = ckpt_frequency
        self.save_top_k = keep_ckpts
        self.monitor = monitor
        self.swa_lrs = swa_lrs

        [self.cbs.append(cb) for cb in callbacks]
        if retain_test_gradients:
            return self.GradientRetainedCallbacks
        return self.Callbacks
