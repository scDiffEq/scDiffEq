# -- import packages: ---------------------------------------------------------
import lightning


# -- callback cls: ------------------------------------------------------------
class BasicProgressBar(lightning.pytorch.callbacks.Callback):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1
        print(f"\nEpoch {current_epoch}/{self.total_epochs} started.")

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1
        print(f"Epoch {current_epoch}/{self.total_epochs} ended.")
