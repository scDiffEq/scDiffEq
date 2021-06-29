class RunningAverageMeter(object):

    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        
import os
import time
import torch
import pandas as pd

def _make_results_tree_structure(adata):
    
    results_folder = adata.uns["run_id"] + "_results"
    
    if os.path.exists(results_folder) == True:
        print("Path exists. Pausing for 5 seconds to give you a chance to interrupt before previous experiment is overwritten.")
        time.sleep(5)
        
    os.makedirs(results_folder, exist_ok=True)
    
    os.makedirs(results_folder + "/validation", exist_ok=True)
    os.makedirs(results_folder + "/model/imgs", exist_ok=True)
    os.makedirs(results_folder + "/training", exist_ok=True)
    os.makedirs(results_folder + "/training/imgs", exist_ok=True)
    os.makedirs(results_folder + "/model", exist_ok=True)
    os.makedirs(results_folder + "/testing", exist_ok=True)
    
    
def _save_validation_loss(adata, results_folder):
        
    validation_loss_savename = results_folder + "/validation/validation_loss.csv"
    validation_iterations = adata.uns["validation_epoch_counter"]
    validation_loss = adata.uns["validation_loss"]
    pd.DataFrame([validation_iterations, validation_loss]).T.to_csv(validation_loss_savename)
    
    return validation_loss_savename
    
def _save_training_loss(adata, results_folder):
    
    
    training_loss_savename = results_folder + "/training/training_loss.csv"
    training_loss = adata.uns["training_loss"]
    training_iterations = range(1, len(list([training_loss])))
    pd.DataFrame([training_iterations, training_loss]).T.to_csv(training_loss_savename)
    
    return training_loss_savename
        
def _save_model(adata, results_folder):   
    
    current_iteration = str("00000") + str(adata.uns["epoch_counter"][-1])    
    model_save_name = results_folder + "/model/model_" + current_iteration
    torch.save(adata.uns["odefunc"].state_dict(), model_save_name)

def save_model_training_statistics(adata):
    
    results_folder = adata.uns["run_id"] + "_results"
    if len(adata.uns["validation_epoch_counter"]) == 1:
        _make_results_tree_structure(adata)
    
    
    training_loss_path = _save_training_loss(adata, results_folder)
    validation_loss_path = _save_validation_loss(adata, results_folder)
    model_vector_field = results_folder + "/model/imgs"
    _save_model(adata, results_folder)
        
    return training_loss_path, validation_loss_path, model_vector_field

### training utilities

def make_run_id_signature(signature_length=10):

    user = getpass.getuser()
    today = date.today().isoformat()
    letters = string.ascii_uppercase
    rand = ''.join(random.choice(letters) for i in range(signature_length))

    signature = "nodescape_" + user + "_" + today + rand

    return signature

def add_run_id(adata, run_id):
    
    if run_id:
        adata.uns["run_id"] = run_id
    else:
        adata.uns["run_id"] = make_run_id_signature()

def preflight(adata, run_id, learning_rate, validation_frequency, plot_smoothing_factor, device):

    """Adds various required components / formatting to adata object. Copies all metadata to AnnData to have one reference for future use"
    
    
    Parameters:
    -----------
    adata
        AnnData object
        
    Returns:
    --------
    
    None
        modifies AnnData object in place. 
    """

    print("Running preflight setup...")
    
    
    if adata.uns["odefunc"]:
        func = adata.uns["odefunc"]
    else:
        print("Please specify a neural network function to be trained")

    # adds adata.uns["run_id"]
    add_run_id(adata, run_id)
    
    adata.uns["number_of_timepoints"] = adata.obs.shape[0] / adata.obs.trajectory.nunique()
    adata.uns["data_dimensionality"] = len(adata.var)
    adata.uns["optimizer"] = optimizer = optim.RMSprop(
        func.parameters(), lr=learning_rate
    )

    adata.uns["training_loss"] = np.array([])
    adata.uns["validation_epoch_counter"] = []
    adata.uns["validation_loss"] = np.array([])

    adata.uns["validation_frequency"] = validation_frequency
    adata.uns["plot_smoothing_factor"] = plot_smoothing_factor
    adata.uns["epoch_counter"] = epoch_counter = 0
    adata.uns["time_meter"] = time_meter = RunningAverageMeter(0.97)
    adata.uns["loss_meter"] = loss_meter = RunningAverageMeter(0.97)

    adata.uns["device"] = device