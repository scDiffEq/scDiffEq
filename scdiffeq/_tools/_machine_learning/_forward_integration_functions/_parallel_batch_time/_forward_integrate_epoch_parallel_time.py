from ._format_parallel_time_batches import _format_parallel_time_batches
from ._forward_integrate_batch import _forward_integrate_batch


def _forward_integrate_epoch_parallel_time(
    self, epoch, time_column="time", n_batches=20
):

    """
    Forward integrates over an epoch of training data (with validation data included). 
    
    Parameters:
    -----------
    adata
        AnnData
        
    epoch
        current epoch of training within `scDiffEq.learn()` module.
        type: int
        
    Returns:
    --------
    None, all is modified in place. 
    
    Notes:
    ------
    (1) Generates new batches each epoch. 
    """

    FormattedBatchedData = _format_parallel_time_batches(
        self, n_batches=n_batches, time_column=time_column, verbose=False
    )

    n_train = FormattedBatchedData["train"][0].batch_y0.shape[0] 
#     * len(
#         FormattedBatchedData["train"].keys()
#     )
    n_valid = FormattedBatchedData["valid"][0].batch_y0.shape[0] 
#     * len(
#         FormattedBatchedData["valid"].keys()
#     )

    epoch_train_loss, epoch_valid_loss = 0, 0

    for [label, batch] in FormattedBatchedData["train"].items():
        epoch_train_loss += _forward_integrate_batch(
            self.adata, batch, device=self.device, validation=False
        )
    self.adata.uns["loss"]["train_loss"].append(epoch_train_loss.item() / n_train)

    if (epoch) % self.adata.uns["validation_frequency"] == 0:
        for [label, batch] in FormattedBatchedData["valid"].items():
            epoch_valid_loss += _forward_integrate_batch(
                self.adata, batch, device=self.device, validation=True
            )
        self.adata.uns["loss"]["valid_loss"].append(epoch_valid_loss / n_valid)
