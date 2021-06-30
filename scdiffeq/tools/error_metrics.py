import torch
import numpy as np

def calc_perc_error(adata, subset):

    """Calculate the percent error for a set of whole trajectories."""
    
    if subset == "test":
        
        try:
            pred_y = adata.uns["latest_test_predictions"].cpu().detach().numpy()
        except:
            pred_y = adata.uns["latest_test_predictions"]
        true_y = adata.uns["latest_test_true_y"].cpu().detach().numpy()
        
        try:
            pred_y = pred_y.reshape(true_y.shape[0], true_y.shape[1], true_y.shape[2])
        except:
            pass
            
        
    elif subset == "training":
        
        try:
            pred_y = adata.uns["latest_training_predictions"].cpu().detach().numpy()
        except:
            pred_y = adata.uns["latest_training_predictions"]
        try:
            true_y = adata.uns["latest_training_true_y"].cpu().detach().numpy()
        except:
            true_y = adata.uns["latest_training_true_y"]
            
    elif subset == "validation":  
        try:
            pred_y = adata.uns["latest_validation_predictions"].cpu().detach().numpy()
        except:
            pred_y = adata.uns["latest_validation_predictions"]
        try:
            true_y = adata.uns["latest_validation_true_y"].cpu().detach().numpy()
        except:
            true_y = adata.uns["latest_validation_true_y"]
        
    percent_error = np.sum(np.abs(pred_y - true_y)) / np.sum(true_y) * 100
            
    return percent_error