
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging


from .. import utils, callbacks

# save_top_k=10,
#     monitor="val_loss",
#     mode="min",
#     dirpath="my/path/",
#     filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
class LightningCallbacksConfiguration(utils.AutoParseBase):
    def __init__(self):
        self.cbs = []

    @property
    def BuiltInCallbacks(self):
        return [
        ModelCheckpoint(every_n_epochs=5,
                        save_top_k=-1,
                       ),
        StochasticWeightAveraging(swa_lrs=1e-2),
        ]        
#         return [ModelCheckpoint(every_n_epochs=10, save_top_k=-1), StochasticWeightAveraging(swa_lrs=1e-2)]

    @property
    def Callbacks(self):
        return self.cbs + self.BuiltInCallbacks

    @property
    def GradientRetainedCallbacks(self):
        return [callbacks.GradientPotentialTest()] + self.cbs

    def __call__(self, callbacks=[], retain_test_gradients=False):

        [self.cbs.append(cb) for cb in callbacks]
        if retain_test_gradients:
            return self.GradientRetainedCallbacks
        return self.Callbacks
