#### IMPORT EXTERNAL PACKAGES
import torch
from torchdiffeq import odeint
import numpy as np

#### IMPORT INTERNAL FUNCTIONS
from .error_metrics import calc_perc_error
from .get_minibatch import get_minibatch
from .sc_odeint import sc_odeint
from ..utilities.subsetting_functions import isolate_trajectory
    
def check_loss_whole_trajectory(adata, validation_data_object, use_embedding):
    
    """"""
    
    iteration=adata.uns["epoch_counter"][-1]
    
    with torch.no_grad():
        validation_batch = get_minibatch(validation_data_object)
        adata.uns["latest_validation_predictions"], validation_loss = sc_odeint(adata, validation_batch, mode="validation", use_embedding=use_embedding)
        
        training_error = calc_perc_error(adata, "training")
        validation_error = calc_perc_error(adata, "validation")
        
        adata.uns["validation_epoch_counter"].append(iteration)
        
        print("Epoch {:04d} || Relative Training Error {:.2f}% || Relative Validation Error {:.2f}%".format(iteration, training_error.item(), validation_error.item()))
                
        return validation_loss
    
    
def check_loss_coarse(adata):
    
    iteration=adata.uns["epoch_counter"][-1]
    
    val_obs = adata.obs.loc[adata.obs.training == True]
    unique_val_trajs = val_obs.trajectory.unique()
    
    with torch.no_grad():
        
        validation_loss = np.array([])

        for traj in unique_val_trajs:

            isolated_traj = isolate_trajectory(adata, traj)
            frac_n = np.round(isolated_traj.shape[0] / 3)

            coarse_array_one_traj = np.array([])

            for _bin in range(1, 4):

                t = (
                    isolated_traj.loc[isolated_traj.coarse_timebin == _bin]
                    .sample(n=int(frac_n))
                    .index.astype(int)
                )
                coarse_array_one_traj = np.append(coarse_array_one_traj, t)

            coarse_array_one_traj = coarse_array_one_traj.reshape(3, int(frac_n))

            # one batch == one traj
            batch_y = []
            batch_pred_y = []

            for i in range(coarse_array_one_traj.shape[1]):

                idx = coarse_array_one_traj[:,i].astype(int)
                y = torch.Tensor(adata.X[idx])
                y0 = y[0]
                t = torch.Tensor(adata.obs.coarse_timebin[idx].values)

                pred_y = odeint(adata.uns["odefunc"], y0, t)
                batch_y.append(y)
                batch_pred_y.append(pred_y)

            batch_y_ = torch.stack(batch_y)
            batch_pred_y_ = torch.stack(batch_pred_y)
            
            adata.uns["latest_validation_true_y"] = batch_y_
            adata.uns["latest_validation_predictions"] = batch_pred_y_

            loss = torch.sum(torch.abs(batch_pred_y_ - batch_y_))
            validation_loss = np.append(validation_loss, loss.item())
            
        validation_loss = np.mean(validation_loss)
        
    if loss !=None:
        adata.uns["validation_loss"] = np.append(adata.uns["validation_loss"], validation_loss)
    else:
        adata.uns["validation_loss"] = np.append(adata.uns["validation_loss"], None)

    training_error = calc_perc_error(adata, "training")
    validation_error = calc_perc_error(adata, "validation")
    adata.uns["validation_epoch_counter"].append(iteration)

    print("Epoch {:04d} || Relative Training Error {:.2f}% || Relative Validation Error {:.2f}%".format(iteration, training_error.item(), validation_error.item()))

    return validation_error.item()