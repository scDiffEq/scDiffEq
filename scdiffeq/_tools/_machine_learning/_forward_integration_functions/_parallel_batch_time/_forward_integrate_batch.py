import torch
from torchdiffeq import odeint


def _get_batch_outs_empty_tensor(batch):

    batch_size = len(batch.batch_y)
    t_steps = len(batch.t)
    n_dim = batch.batch_y.shape[-1]

    return (
        torch.zeros([batch_size, t_steps, n_dim]),
        torch.zeros([batch_size, t_steps, n_dim]),
    )


def _forward_integrate_batch(
    adata, batch, device, rtol=1e-7, atol=1e-9, method="dropri5", validation=False
):

    """
    Forward integrates one batch for training / validation. 
    Assumes training unless validation is indicated. 
    
    Parameters:
    -----------
    adata
    
    batch_of_trajectories
    
    validation
    
    Returns:
    --------
    batch_loss
    
    Notes:
    ------
    """

    y0 = batch.batch_y0.reshape(batch.batch_y0.shape[0], 1, batch.batch_y0.shape[1])

    if not validation:
        adata.uns["optimizer"].zero_grad()

    if validation:
        with torch.no_grad():
            y_pred = odeint(
                adata.uns["ODE"], y0, batch.t, rtol=rtol, atol=atol, method=method
            ).to(device)

    else:
        y_pred = odeint(adata.uns["ODE"], y0, batch.t).to(device)

    batch_loss = adata.uns["loss_func"](y_pred, batch.batch_y)

    if not validation:
        batch_loss.backward()
        adata.uns["optimizer"].step()

    return batch_loss
