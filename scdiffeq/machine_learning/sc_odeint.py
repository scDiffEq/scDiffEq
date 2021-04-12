import torch
import numpy as np
from torchdiffeq import odeint

def _check_increasing_time_minibatch(minibatch, i):

    """I can't figure out why, but sometimes, after the window is generated, t still comes out with duplicates. This function will be a bandaid until I figure that out."""
    if len(np.unique(minibatch[i].t)) == len(minibatch[i].t):

        return True
    
def sc_odeint(adata, minibatch, mode):
    
    """
    The main ML function for the original neural ODEs pytorch implementation. 
    This is used for whole-trajectory training. Made available as a user-facing
    function but employed by `sdq.ml.train_model` and batching is required to be
    performed to get the data in the correct format. 
    
    Parameters:
    -----------
    adata
        AnnData object.
    
    minibatch
        In practice, a data subset. If the dataset is small, this will be an entire epoch. 
    
    mode
        "train" or "validation"
    
    Returns:
    --------
    predicted_y, loss
    """
    
    predicted_y_minibatch = []
    minibatch_y = []

    for i in range(len(minibatch)):
        
        if _check_increasing_time_minibatch(minibatch, i) == True:
                
            predicted_y = odeint(adata.uns["odefunc"], minibatch[i].y0, minibatch[i].t, method="euler").to(adata.uns["device"])
            predicted_y_minibatch.append(predicted_y)
            minibatch_y.append(minibatch[i].y)
        
    predicted_y_minibatch = torch.stack(predicted_y_minibatch)
    minibatch_y = torch.stack(minibatch_y)        
    loss = torch.sum(torch.abs(predicted_y_minibatch - minibatch_y))
    
    if mode == "train":
        loss.backward()

        adata.uns["latest_training_true_y"] = minibatch_y

        if loss != None:
            adata.uns["loss_meter"].update(loss.item())
            adata.uns["training_loss"] = np.append(adata.uns["training_loss"], loss.item())
        else:
            adata.uns["training_loss"] = np.append(adata.uns["training_loss"], None)

    if mode == "validation":

        adata.uns["latest_validation_true_y"] = minibatch_y

        if loss !=None:

            adata.uns["validation_loss"] = np.append(adata.uns["validation_loss"], loss.item())
        else:
            adata.uns["validation_loss"] = np.append(adata.uns["validation_loss"], None)

    return predicted_y_minibatch, loss