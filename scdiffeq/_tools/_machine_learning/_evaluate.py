
# package imports #
# --------------- #
import torch

from ._forward_integration_functions._format_trajectories_into_batches import _format_trajectories
from ._forward_integration_functions._forward_integrate_one_trajectory import _forward_integrate_one_trajectory

def _get_batch_outs_empty_tensor(batch_of_trajectories):
    
    for key in batch_of_trajectories.keys():
        break
    
    batch_size = len(batch_of_trajectories)
    traj_shape = batch_of_trajectories[key].data.shape
    t_steps, n_dim = traj_shape[0], traj_shape[1]

    return torch.zeros([batch_size, t_steps, n_dim]), torch.zeros(
        [batch_size, t_steps, n_dim]
    )

def _format_test_trajectories(adata):

    test_data = adata.uns["data_split_keys"]["test"]
    test_trajs = _format_trajectories(test_data)

    return test_trajs


def _evaluate(adata):

    """
    Evaluate model on test data.

    Parameters:
    -----------

    Returns:
    --------

    Notes:
    ------
    """

    
    adata.uns["test_accuracy"] = 0
    adata.uns['test_trajectories'] = test_trajectories = _format_test_trajectories(adata)
    adata.uns["test_y_predicted"], adata.uns["test_y"] = _get_batch_outs_empty_tensor(test_trajectories)

   
    for n, trajectory in enumerate(test_trajectories.keys()):

        with torch.no_grad():
            (
                adata.uns["test_y_predicted"][n],
                adata.uns["test_y"][n],
            ) = _forward_integrate_one_trajectory(adata.uns['ODE'], test_trajectories[trajectory])
            adata.uns["test_accuracy"] += adata.uns['loss_func'](
                adata.uns["test_y_predicted"][n], adata.uns["test_y"][n]
            )
    print("MSELoss: {:.4f}".format(adata.uns["test_accuracy"].item() / (len(test_trajectories))))