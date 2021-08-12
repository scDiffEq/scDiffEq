
from ._format_trajectories_into_batches import _get_formatted_training_validation_trajectories
from ._forward_integrate_batch import _forward_integrate_batch

def _forward_integrate_epoch(adata, epoch):
    
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

    batches = _get_formatted_training_validation_trajectories(adata, n_batches=10)
    
    n_train_trajs_ = len(batches["train_batches"]) * len(batches["train_batches"][0])
    n_valid_trajs_ = len(batches["valid_batches"]) * len(batches["valid_batches"][0])
    
    epoch_train_loss, epoch_valid_loss = 0, 0
    
    for [label, batch] in batches["train_batches"].items():
        epoch_train_loss += _forward_integrate_batch(adata, batch, validation=False)
    adata.uns["loss"]["train_loss"].append(epoch_train_loss.item() / n_train_trajs_)
        
    if (epoch) % adata.uns["validation_frequency"] == 0:
        for [label, batch] in batches["valid_batches"].items():
            epoch_valid_loss += _forward_integrate_batch(adata, batch, validation=True)
        adata.uns["loss"]["valid_loss"].append(epoch_valid_loss / n_valid_trajs_)
    
    
    