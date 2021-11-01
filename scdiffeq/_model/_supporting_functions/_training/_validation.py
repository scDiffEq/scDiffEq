import torch
import numpy as np
import matplotlib.pyplot as plt
import vintools as v
import os
from IPython.display import display

from ._plot_training_validation import _plot_validation_training_update
from .._common_functions._IntegratorModule import _Integrator

def _validate(adata, 
              network_model, 
              diffusion, 
              integration_function, 
              HyperParameters, 
              TrainingMonitor,
              use,
              time_key,
              plot,
             ):

    """"""
    
    validator = _Integrator(
        adata=adata,
        mode='valid',
        network_model=network_model,
        diffusion=diffusion,
        integration_function=integration_function,
        HyperParameters=HyperParameters,
        TrainingMonitor=TrainingMonitor,
        use=use,
        time_key=time_key,
    )

    validator.forward_integrate()
    validator.calculate_loss()
    
    # we don't need to redo the training forward_int for the plot - we can just use the last score stored in the TrainingMonitor

#     if plot:
#         _plot_validation_training_update(validator,
#                          TrainingMonitor, 
#                          HyperParameters,
#                          binsize=10, 
#                          title_fontsize=16, 
#                          save_path=plot_savepath)

#     print("Validation loss: {:.4f}".format(validator.valid_loss))