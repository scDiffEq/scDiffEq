import os
import glob
import numpy as np

def _get_best_ckpt(version_ckpts):
    
    val_losses = []
    for v in version_ckpts:
        if not os.path.basename(v).split(".")[0] == "last":
            val_loss = float(os.path.basename(v).split("=")[-1].split(".ckpt")[0])
            val_losses.append(val_loss)
    best_loss = np.min(val_losses)

    for v in version_ckpts:
        if str(best_loss) in v:
            return v
        
def _get_best_epoch_each_version(model_path, verbose=True):
    
    """
    For an individual model (specified by the path to the model_outs, return
    the best available epoch ckpts for each seed / version of the model.
    
    
    Parameters:
    -----------
    model_path
    
    Returns:
    --------
    best_ckpt_dict
        type: dict
    """
    
    best_ckpt_dict = {}
    version_paths = glob.glob(model_path + "/v*")
    for version_path in version_paths:
        version_ckpts = glob.glob(version_path + "/*")
        best_ckpt_path = _get_best_ckpt(version_ckpts)
        best_ckpt_dict[os.path.basename(version_path)] = best_ckpt_path
    
    if verbose:
        print("Found the following:\n")
        for k, v in best_ckpt_dict.items():
            print("{}: {}".format(k, v))
    
    return best_ckpt_dict