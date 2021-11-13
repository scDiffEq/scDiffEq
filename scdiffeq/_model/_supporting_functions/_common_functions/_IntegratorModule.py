
# _IntegratorModule.py
__module_name__ = "_IntegratorModule.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])


# package imports #
# --------------- #
import torch


# local imports #
# ------------- #
from ._fetch_data_from_adata import _fetch_adata
from ._reshape_compatible import _reshape_compatible
from ._create_batches import _create_batches

from ...._utilities._format_string_printing_font import _format_string_printing_font

def _forward_integrate(integration_function, parallel, diffusion, network_model, y0, t, device):

    """"""

    if parallel and not diffusion:

        pred_y = integration_function(network_model.f, y0, t).to(device)

    elif parallel and diffusion:
        pred_y = integration_function(network_model, y0, t).to(device)
    else:
        print(_format_string_printing_font("Invalid network_model passed.", ["BOLD", "RED"]))

    return pred_y

class _Integrator:
    def __init__(
        self,
        mode,
        network_model,
        device,
        diffusion,
        integration_function,
        HyperParameters,
        TrainingMonitor,
        use="X",
        time_key="time",
    ):

        """
        DiffEq is passed, shares some details with this class and then is not saved.


        Parameters:
        -----------
        mode

        Notes:
        ------
        (1) Shared between Evalution, Validation, and Learning.
        (2) parellel=True is currently the only data split mode available.
        (3) `mode` is the most important parameter.
        (4) It is potentially useful to keep `mode` and `group` separate despite the redundancy
            in a non-edge case such that one could test/train/validate on an alternative group if desired.
        """
        
        self.device=device
        self.use = use
        self.time_key = time_key
        self.mode = mode
        self.TrainingMonitor = TrainingMonitor
        self.integration_function = integration_function
        self.hypers = HyperParameters
        self.loss_function = self.hypers.loss_function
        self.optimizer = self.hypers.optimizer
        self.network_model = network_model
        self.diffusion = diffusion
        self.parallel = True
        
    def batch_data(self, adata, n_batches):
        adata_group = adata[adata.obs[self.mode]].copy()
        adata_group.obs = adata_group.obs.reset_index(drop=True)
                
        self.BatchAssignments = _create_batches(adata_group, n_batches)
        self.BatchedData = {}
            
        for batch in self.BatchAssignments.keys():
            self.BatchedData[batch] = {}            
            self.BatchedData[batch]["adata"] = adata_group[
                adata_group.obs.loc[
                    adata_group.obs.trajectory.isin(self.BatchAssignments[batch])
                ].index.astype(int)
            ]
            self.BatchedData[batch]["adata"].obs = self.BatchedData[batch]["adata"].obs.reset_index(drop=True)

            if self.BatchedData[batch]["adata"].shape[0] > 0:
                
                (
                    self.BatchedData[batch]["y"],
                    self.BatchedData[batch]["y0"],
                    self.BatchedData[batch]["t"],
                    self.BatchedData[batch]["batch_size"],
                ) = _fetch_adata(self.BatchedData[batch]["adata"], 
                                 self.network_model, 
                                 self.device, 
                                 self.use, 
                                 self.time_key
                )
                
    def forward_integrate(self):
        
        

        if self.mode == "train":
            self.optimizer.zero_grad()
            self.BatchedPredictions = {}
            for batch in self.BatchedData.keys():
                self.network_model.batch_size = self.BatchedData[batch]["batch_size"]
                self.BatchedPredictions[batch] = _forward_integrate(self.integration_function,
                                                 self.parallel, 
                                                 self.diffusion, 
                                                 self.network_model,
                                                 self.BatchedData[batch]["y0"],
                                                 self.BatchedData[batch]["t"],
                                                 self.device,
            )
        else:
            with torch.no_grad():
                self.BatchedPredictions = {}
                for batch in self.BatchedData.keys():
                    self.network_model.batch_size = self.BatchedData[batch]["batch_size"]
                    self.BatchedPredictions[batch] = _forward_integrate(self.integration_function,
                                                         self.parallel, 
                                                         self.diffusion, 
                                                         self.network_model,
                                                         self.BatchedData[batch]["y0"],
                                                         self.BatchedData[batch]["t"],
                                                         self.device,
                )

    def calculate_loss(self):
                
        self.loss = torch.zeros(len(self.BatchedData.keys()))
        
        for batch in self.BatchedData.keys():
            self.network_model.batch_size = self.BatchedData[batch]["batch_size"]
            
            self.BatchedPredictions[batch] = _reshape_compatible(self.BatchedPredictions[batch])
            self.loss[batch] = self.loss_function(
                    self.BatchedPredictions[batch], self.BatchedData[batch]['y'].reshape(self.BatchedPredictions[batch].shape)
                )
        
        self.loss = self.loss.sum()
                    
        if self.mode == "valid":
            self.TrainingMonitor.update_loss(self.loss.item(), validation=True)
        elif self.mode == "train":
            self.loss.backward()
            self.optimizer.step()
            self.TrainingMonitor.update_loss(self.loss.item())
        elif self.mode == "test":
            self.test_loss = self.loss.item()
        else:
            print("Invalid forward integration mode")