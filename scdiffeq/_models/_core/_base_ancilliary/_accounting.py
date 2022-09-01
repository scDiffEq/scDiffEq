
import torch

def _count_params(model):

    """
    Parameters:
    -----------
    model
    
    Returns:
    --------
    param_counts
    
    Notes:
    ------
    (1) There may be a better / native way to do this in pytorch lightning
    """

    param_counts = {"total": 0, "trainable": 0, "non-trainable": 0}

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_counts["trainable"] += parameter.numel()
        else:
            param_counts["non-trainable"] += parameter.numel()
        param_counts["total"] += parameter.numel()

    return param_counts


def _retain_gradients_for_potential(model):
    if model.forward_function.using_potential_net:
        torch.set_grad_enabled(True)
        model.train()
        
        
def _update_loss_logs(model, loss, t, epoch, batch_idx, stage="train", metric="sinkhorn", time_unit="d"):

    for i in range(1, len(loss)):
        log_description = "{}.{}_loss.{}{}".format(stage, metric, time_unit, int(t[i]))
        model.log(log_description, loss[i].item())