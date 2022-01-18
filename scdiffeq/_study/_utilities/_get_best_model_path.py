

import glob
import numpy as np
import os

class _BestModel:

    def __init__(self, path, epoch):

        self.path = path
        self.epoch = epoch

def _get_best_model_path(path):

    """
    Returns:
    --------
    BestModel
       two sub-classes:
        (1) the path to the best model
        (2) the epoch at which the best model has been observed. 
    """
        
    model_paths = glob.glob(os.path.join(path, "best.model.epoch_*"))
    best_model_epochs = []

    for mp in model_paths:
        best_model_epochs.append(mp.split("_")[1])

    best_model_path = model_paths[np.array(best_model_epochs).astype(int).argmax()]
    best_model_epoch = int(best_model_path.split("_")[1])
    
    BestModel = _BestModel(best_model_path, best_model_epoch)

    return BestModel