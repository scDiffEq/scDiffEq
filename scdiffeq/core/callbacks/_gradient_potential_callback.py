

from lightning import Callback


class GradientPotentialTest(Callback):
    def on_validation_batch_end(
        self, trainer, model, outputs, batch, batch_idx, dl_idx
    ):

        test_loss = outputs["loss"].item()
        print("Test loss: {}".format(test_loss))
