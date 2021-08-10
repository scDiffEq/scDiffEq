
from ._error_metrics import _calc_perc_error
from ..utilities._torch_device import _set_device
from ..utilities._subsetting_functions import _group_adata_subset
from ..plotting._plot_predictions import _plot_predictions
from ..plotting._plot_vectorfield import _plot_vectorfield

from torchdiffeq import odeint
from sklearn.decomposition import PCA
import torch
import numpy as np

def _isolate_trajectory(data_object, trajectory_number):

    """
    Get the entire obs df for a given trajectory.
    
    Parameters:
    -----------
    data_object
        assumes the the object contains data_object.obs
    
    trajectory_number
        number that is assumed to be present within data_object.obs.trajectory
        
    Returns:
    --------
    df
        data_object.obs dataframe subset for only the trajectory queried
    """

    df = data_object.obs.loc[data_object.obs.trajectory == trajectory_number]

    return df


def _evaluate_test_traj(adata, device='cpu', data_subset="test", time_name="time", plot=False, plot_savename=None):

    """
    
    """

    try:
        adata.X = adata.X.toarray()
    except:
        pass

    predictions = []
    true_y = []

    data_object_subset = _group_adata_subset(adata, data_subset, time_name=time_name)
    data_object_subset.obs = data_object_subset.obs.reset_index()
    trajectories = data_object_subset.obs.trajectory.unique()

    for i in trajectories:
        traj_df = _isolate_trajectory(data_object_subset, i)

        y = torch.Tensor(
            data_object_subset.data[traj_df.index.values.astype(int)]  # .toarray()
        )
        y0 = torch.Tensor(y[0])
        t = torch.Tensor(traj_df.time.values)
        
        if device != 'cpu':
            device = 'cuda:0'
        else:
            pass
        
        if len(np.unique(t.cpu())) == len(t.cpu()):
            device=set_device(gpu='0')
            predicted_y = odeint(adata.uns["odefunc"].to(device), y0.to(device), t.to(device)).to(device)
            predictions.append(predicted_y)
            true_y.append(y)
            
    predictions_ = torch.stack(predictions)
    true_y = torch.stack(true_y)
    
    adata.uns["latest_test_true_y"] = true_y

    adata.uns["num_predicted_trajectories"] = predictions_.shape[0]
    predictions_ = predictions_.reshape(
        predictions_.shape[0] * predictions_.shape[1], predictions_.shape[2]
    )
    
    predictions_ = predictions_.cpu().detach().numpy()
#     adata.uns["true_y"] = true_y
    adata.uns["predictions"] = predictions_
    adata.uns["latest_test_predictions"] = predictions_
    if predictions_.shape[1] > 2:
        print("Dimensional reduction...")
        pca = adata.uns["pca"]
        pcs = pca.fit_transform(predictions_)

        adata.uns["predictions_pca"] = pcs
    else:
        pass 
    print(adata)
    
    if plot == True:
        
        _plot_predictions(adata, subset=data_subset, savename=plot_savename)
        
        if predictions_.shape[1] == 2:
            _plot_vectorfield(adata, savename=plot_savename)
    test_error = calc_perc_error(adata, "test")  
    print("Relative error: {:.2f}%".format(test_error))