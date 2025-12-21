# -- import packages: ---------------------------------------------------------
import lightning
import logging
import os
import pickle

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)


# -- callback cls: ------------------------------------------------------------
class IntermittentSaves(lightning.pytorch.callbacks.Callback):
    def __init__(self, frequency: int = 50) -> None:
        """
        Args:
            frequency (int): save frequency
        """

        super().__init__()

        self.frequency = frequency

    def on_train_batch_end(
        self, trainer, model, outputs, batch, batch_idx, *args, **kwargs
    ):
        """
        Args:
            trainer (lightning.pytorch.Trainer): trainer
            model (lightning.pytorch.LightningModule): model
            outputs (dict): outputs
            batch (torch.Tensor): batch
            batch_idx (int): batch index
        """

        epoch = model.current_epoch

        if (epoch % self.frequency == 0) and (batch_idx == 0):
            dpath = os.path.join(trainer.logger.log_dir, "ckpt_outputs")
            if not os.path.exists(dpath):
                os.mkdir(dpath)
                logger.info(f"Created directory: {dpath}")

            fpath = os.path.join(
                dpath, "epoch_{}.batch_{}.pkl".format(epoch, batch_idx)
            )
            pickle.dump(
                obj=outputs,
                file=open(fpath, "wb"),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
