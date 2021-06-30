
#### IMPORT EXTERNAL PACKAGES
import os
import time
import torch
import string
import random
import getpass
import numpy as np
from datetime import date
import torch.optim as optim

#### IMPORT INTERNAL FUNCTIONS
from ..utilities.torch_device import set_device
from ..utilities.save_adata import write_h5ad
from ._ml_utils import RunningAverageMeter
from .validation import check_loss_whole_trajectory as check_loss
from .get_minibatch import get_minibatch
from .sc_odeint import sc_odeint
from ._ml_utils import save_model_training_statistics as save_model

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..plotting.plotting_presets import single_fig_presets as presets




def running_mean(x, N):
    
    x = [float(i) for i in x]

    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_training_validation_loss(adata, training_loss_path, validation_loss_path):
    
    N = adata.uns["validation_frequency"]
    test_frequency = adata.uns["plot_smoothing_factor"]
    
    
    train_norm_factor = adata.obs.training.sum() / adata.uns["number_of_timepoints"]
    valid_norm_factor = adata.obs.validation.sum() / adata.uns["number_of_timepoints"]
    
    ### load training and val saved data .csvs
    
    train_df = pd.read_csv(training_loss_path, index_col=0, names=["Iteration", "Training Error"], skiprows=1)
    val_df = pd.read_csv(validation_loss_path, index_col=0, names=["Iteration", "Validation Error"], skiprows=1)
    
    ### calculate running averages
    
    ### actual plotting
    presets(title="Training and Validation Loss", x_lab="Epoch", y_lab="Error", size=(10, 8))
    
    if train_df.shape[0] > N:
        
        plt.scatter(
            x=train_df["Iteration"],
            y=(train_df["Training Error"].values.astype(float) / train_norm_factor)*100,
            s=10,
            c="mediumpurple",
            alpha=0.5,
            label = "Training"
        )

        train_running_avg = running_mean(train_df["Training Error"].values.astype(float), N)

        plt.plot((train_running_avg / train_norm_factor)*100, linewidth=3, c="mediumpurple") #, label="Training MAE Smoothed")
    
        plt.scatter(
            x=val_df["Iteration"],
            y=(val_df["Validation Error"].values.astype(float) / valid_norm_factor)*100,
            s=10,
            c="orange",
            alpha=0.5,
            label = "Validation"
        )

    
    if val_df.shape[0] > N:
        valid_running_avg = running_mean(val_df["Validation Error"].values.astype(float), N)
        validation_x_axis = np.array(range(1,len(valid_running_avg)+1))*test_frequency
        
        plt.plot(validation_x_axis , (valid_running_avg / valid_norm_factor)*100, linewidth=3, c="orange") # , label="Validation MAE Smoothed")
    train_img_path = os.path.join(training_loss_path.split("/")[:-1][0], training_loss_path.split("/")[:-1][1])
    training_imgs_savename = train_img_path + "/imgs/training_validation_progress" + str(max(train_df["Iteration"].values)) + ".png"
    plt.legend(
        markerscale=3,
        edgecolor="w",
        fontsize=14,
        handletextpad=None,
        bbox_to_anchor=(0.5, 0.0, 0.70, 1),
    )
    plt.savefig(training_imgs_savename)
    plt.show() 
    
    

#### IMPORT WHOLE TRAJECTORY FUNCTIONS

def make_run_id_signature(signature_length=10):

    user = getpass.getuser()
    today = date.today().isoformat()
    letters = string.ascii_uppercase
    rand = ''.join(random.choice(letters) for i in range(signature_length))

    signature = "_".join(["scdiffeq", user, today, rand])

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
    
def save(adata, plot_training):
    
    
    training_loss_path, validation_loss_path, model_vector_field = save_model(adata)
    
    if plot_training==True:
        plot_training_validation_loss(adata, training_loss_path, validation_loss_path)
            
    write_h5ad(adata)
    
    
def plot_each_gene(adata, columns=2):
    
    down = round(len(adata.var) / columns)
    fig, axs = plt.subplots(columns, down)

    count = 0

    for i in range(columns):
        for j in range(down):
            gene_pred = adata.uns["latest_training_predictions"][:, :, count].T
            gene_adata = adata[:, count]
            axs[i, j].plot(gene_adata.X, c="blue")
            axs[i, j].plot(gene_pred.detach().numpy(), c="orange")
            count += 1
            
    plt.show()
    
def train_model(
    adata,
    data_object,
    run_id=None,
    learning_rate=1e-03,
    epochs=1,
    validation_frequency=20,
    device=set_device(),
    plot_training=False,
    plot_smoothing_factor=3,
    use_embedding=False,
):

    """
    Use this function to train a nueral ODE model. 
    Made for use with AnnData.
    Uses a torch neural network input.
    Data is formatted into objects.
    
    Parameters:
    -----------
    adata
        AnnData object
        
    data_object
        Formatted using the function n.ml.split_test_train(adata)
    
    run_id
        identifier for savenames related to the experiment
        default=None
    
    learning_rate
        default=1e-03    
    
    epochs
        number of epochs over which to train the model
        default=1
        
    validation_frequency
        At every epoch divisible by the validation_frequency, validation is performed.
        default=20
    
    device
        default=set_device()
    
    plot_training
        default=False
    
    plot_smoothing_factor
        default=3
    
    Returns:
    --------
    None
        AnnData is updated in place.
        
    """
    
    preflight(adata, run_id, learning_rate, validation_frequency, plot_smoothing_factor, device)
    
    print("Preflight complete. Beginning training...")
    print("")
    print("Epoch | Loss")
    print("------|--------")
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        
        adata.uns["optimizer"].zero_grad()
        batch = get_minibatch(data_object.train)
        adata.uns["latest_training_predictions"],training_loss = sc_odeint(adata, batch, mode="train", use_embedding=use_embedding)
        adata.uns["optimizer"].step()
        adata.uns["time_meter"].update(time.time() - t0)
        adata.uns["epoch_counter"] = range(1, epoch + 1)
        
        if epoch % validation_frequency == 0:
            
            if epoch >= 1000:
                print(epoch, " |", '{0:.4f}'.format(training_loss.item()))
            elif epoch >= 100:
                print(epoch, "  |", '{0:.4f}'.format(training_loss.item()))
            else:
                print(epoch, "   |", '{0:.4f}'.format(training_loss.item()))
        
        if epoch % 500 == 0:
            plot_each_gene(adata)
            
            
            
#         if epoch % validation_frequency == 0:
#             validation_loss = check_loss(adata, data_object.validation, use_embedding=use_embedding)
#             save(adata, plot_training)
            
#             t = time.time()
#             current = (t - t0) / 3600

#             print("Time elapsed:", str(current), "h")
#             print("Average time per epoch:", str(current / adata.uns["epoch_counter"][-1]), "h")