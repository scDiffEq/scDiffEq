# _IntegratorModule.py
__module_name__ = "_IntegratorModule.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(
    [
        "vinyard@g.harvard.edu",
    ]
)

# package imports #
# --------------- #
import vintools as v
import torch

# local imports #
# ------------- #
from ._fetch_data_from_adata import _fetch_adata
from ._reshape_compatible import _reshape_compatible


def _forward_integrate(integration_function, parallel, diffusion, network_model, y0, t):

    """"""

    if parallel and not diffusion:

        pred_y = integration_function(network_model.f, y0, t)

    elif parallel and diffusion:
        pred_y = integration_function(network_model, y0, t)
    else:
        print(v.ut.format_pystring("Invalid network_model passed.", ["BOLD", "RED"]))

    return pred_y

class Integrator:
    def __init__(
        self,
        adata,
        mode,
        group,
        network_model,
        diffusion,
        integration_function,
        hyper_parameters,
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

        self.mode = mode
        self.TrainingMonitor = TrainingMonitor
        self.integration_function = integration_function
        self.hypers = hyper_parameters
        self.loss_function = self.hypers.loss_function
        self.optimizer = self.hypers.optimizer
        self.network_model = network_model
        self.diffusion = diffusion
        self.parallel = True
        self.y, self.y0, self.t = _intake_adata(
            adata, mode, network_model, use, time_key
        )

    def forward_integrate(self):

        if self.mode == "train":
            self.optimizer.zero_grad()
            self.pred_y = _forward_integrate(self.integration_function,
                                             self.parallel, 
                                             self.diffusion, 
                                             self.network_model,
                                             self.y0,
                                             self.t
            )
        else:
            with torch.no_grad():
                self.pred_y = _forward_integrate(self.integration_function,
                                                 self.parallel, 
                                                 self.diffusion, 
                                                 self.network_model,
                                                 self.y0,
                                                 self.t
                )

    def calculate_loss(self):

        self.pred_y = _reshape_compatible(self.pred_y)
        self.loss = self.loss_function(
            self.pred_y, self.y.reshape(self.pred_y.shape)
        ).item()

        if mode == "valid":
            self.TrainingMonitor.update_loss(self.loss, validation=True)
        elif mode == "train":
            loss.backward()
            self.optimizer.step()
            self.TrainingMonitor.update_loss(loss.item())
        elif mode == "test":
            self.test_loss = self.loss
        else:
            print("Invalid forward integration mode")