
import torch

from ._forward_integrate_one_trajectory import _forward_integrate_one_trajectory

def _get_batch_outs_empty_tensor(batch_of_trajectories):

    batch_size = len(batch_of_trajectories)
    traj_shape = batch_of_trajectories[0].data.shape
    t_steps = traj_shape[0]
    n_dim = traj_shape[1]

    return torch.zeros([batch_size, t_steps, n_dim]), torch.zeros(
        [batch_size, t_steps, n_dim]
    )


def _forward_integrate_batch(adata, batch_of_trajectories, validation=False):

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
       
    if not validation:
        adata.uns['optimizer'].zero_grad()

    y_pred, y_data = _get_batch_outs_empty_tensor(batch_of_trajectories)

    for n_traj, trajectory in enumerate(batch_of_trajectories):

        if validation:
            with torch.no_grad():
                y_pred[n_traj], y_data[n_traj] = _forward_integrate_one_trajectory(
                    adata.uns['ODE'], trajectory
                )
        else:
            y_pred[n_traj], y_data[n_traj] = _forward_integrate_one_trajectory(
                adata.uns['ODE'], trajectory
            )
            
    batch_loss = adata.uns["loss_func"](y_pred, y_data)

    if not validation:
        batch_loss.backward()
        adata.uns['optimizer'].step()
        
    return batch_loss