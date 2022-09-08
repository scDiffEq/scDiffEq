

from pytorch_lightning.callbacks import ModelCheckpoint

def timepoint_recovery_checkpoint(
    log_path,
    filename="{epoch}-timepoint_recovery-{val_loss_d6:.2f}",
    monitor="val_loss_d6",
    save_top_k=5,
    every_n_epochs=1,
    **kwargs
):

    """
    This is the model checkpoint strategy for the timepoint
    recovery task.
    """

    return ModelCheckpoint(
        dirpath=log_path,
        filename=filename,
        monitor=monitor,
        save_last=True,
        save_top_k=save_top_k,
        every_n_epochs=every_n_epochs,
        **kwargs,
    )