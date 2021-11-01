
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm
from IPython.display import clear_output

from ._validation import _validate
from .._common_functions._IntegratorModule import _Integrator

def _learn_diffeq(adata, 
                  network_model,
                  n_batches,
                  device,
                  diffusion, 
                  integration_function, 
                  HyperParameters, 
                  TrainingMonitor,
                  validation_status,
                  use,
                  time_key,
                  plot,
                 ):

    """"""
        
    TrainingMonitor.start_timer()
        
    learner = _Integrator(mode='train',
                          network_model=network_model,
                          device=device,
                          diffusion=diffusion,
                          integration_function=integration_function,
                          HyperParameters=HyperParameters,
                          TrainingMonitor=TrainingMonitor,
                          use=use,
                          time_key=time_key,
                     )
    validator = _Integrator(mode='valid',
                            network_model=network_model,
                            device=device,
                            diffusion=diffusion,
                            integration_function=integration_function,
                            HyperParameters=HyperParameters,
                            TrainingMonitor=TrainingMonitor,
                            use=use,
                            time_key=time_key,
                        )
    
    for epoch_n in tqdm(range(1, int(HyperParameters.n_epochs + 1))):
        learner.batch_data(adata, n_batches)
        learner.forward_integrate()
        learner.calculate_loss()
        TrainingMonitor.update_time()
        TrainingMonitor.current_epoch += 1
        
        if validation_status and (epoch_n % HyperParameters.validation_frequency == 0):
#             print("Elapsed training time: {}s".format(TrainingMonitor.elapsed_time))
#             clear_output(wait=True)
            validator.batch_data(adata, n_batches)
            validator.forward_integrate()
            validator.calculate_loss()
        
    print("Total training time: {}s".format(TrainingMonitor.elapsed_time))
    return learner