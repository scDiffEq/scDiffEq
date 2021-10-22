
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm

def _get_y0_idx(df, time_key):

    """"""

    y0_idx = df.index[np.where(df[time_key] == df[time_key].min())]

    return y0_idx


def _get_adata_y0(adata, time_key):

    y0_idx = _get_y0_idx(adata.obs, time_key)
    adata_y0 = adata[y0_idx].copy()

    return adata_y0


def _get_y0(adata, use, time_key):

    adata_y0 = _get_adata_y0(adata, time_key)

    if use == "X":
        return torch.Tensor(adata_y0.X)

    elif use in adata.obsm_keys():
        return torch.Tensor(adata_y0.obsm[use])

    else:
        print("y0 not properly defined!")


def _fetch_data(adata, use="X", time_key="time"):

    """

    Assumes parallel time.
    """

    y = torch.Tensor(adata.X)
    y0 = _get_y0(adata, use, time_key)
    t = torch.Tensor(adata.obs[time_key].unique())

    return y, y0, t

def _shape_compatible(pred_y):
    
    """"""
    
    reshaped_outs = []
    for i in range(pred_y.shape[1]):
        reshaped_outs.append(pred_y[:, i, :])
        
    return torch.stack(reshaped_outs)


class Learner:

    """"""

    def __init__(
        self, 
        adata, 
        network_model, 
        diffusion,
        integration_function, 
        hyper_parameters, 
        TrainingMonitor, 
        use="X", 
        time_key="time", 
        parallel=True
    ):

        """DiffEq is passed, shares some details with this class and then is not saved."""

        self.integration_function = integration_function
        self.parallel = parallel
        self.hypers = hyper_parameters
        self.loss_function = self.hypers.loss_function
        self.optimizer = self.hypers.optimizer
        self.network_model = network_model
        self.diffusion = diffusion
        self.y, self.y0, self.t = _fetch_data(adata, use, time_key)
        self.TrainingMonitor = TrainingMonitor

    def forward_integrate(self):

        self.optimizer.zero_grad()
        if self.parallel and self.diffusion:
            print("Training neural SDE in parallel time mode.")
                
        if self.parallel and not self.diffusion:
            self.pred_y = self.integration_function(
                self.network_model.f, self.y0, self.t
            ) # .reshape([3, 50, 2])

    def calculate_loss(self):
        
        self.pred_y = _shape_compatible(self.pred_y)
        loss = self.loss_function(self.pred_y, self.y.reshape(self.pred_y.shape))
        loss.backward()
        self.optimizer.step()
        self.TrainingMonitor.update_loss(loss.item())
        
from IPython.display import clear_output        
from ._validation import _validate

def _learn_diffeq(adata, 
                  network_model, 
                  diffusion, 
                  integration_function, 
                  HyperParameters, 
                  TrainingMonitor,
                  validation_status,
                  plot,
                  valid_plot_savepath):

    """"""
    
    train_adata = adata[adata.obs["train"]]
    TrainingMonitor.start_timer()
    learner = Learner(train_adata, 
                      network_model, 
                      diffusion,
                      integration_function, 
                      HyperParameters,
                      TrainingMonitor,
                     )
    
    for epoch_n in tqdm(range(1, int(HyperParameters.n_epochs + 1))):
        learner.forward_integrate()
        learner.calculate_loss()
        TrainingMonitor.update_time()
        TrainingMonitor.current_epoch += 1
        
        
        if validation_status and (epoch_n % HyperParameters.validation_frequency == 0):
            print("Elapsed training time: {}s".format(TrainingMonitor.elapsed_time))
            clear_output(wait=True)
            _validate(adata, 
                      network_model, 
                      diffusion, 
                      integration_function, 
                      HyperParameters, 
                      TrainingMonitor,
                      plot,
                      valid_plot_savepath)
            
    print("Total training time: {}s".format(TrainingMonitor.elapsed_time))
    return learner