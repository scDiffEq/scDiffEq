
__module_name__ = "_BaseModel.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
from pytorch_lightning import LightningModule
import torchsde # will be removed eventually
import torch
import time
import numpy as np


# import local dependencies #
# ------------------------- #
from ._ancilliary._WassersteinDistance import WassersteinDistance
from ._ancilliary._shape_tools import _restack_x
from ._ancilliary._count_params import count_params
from ._ancilliary._ForwardFunctions import ForwardFunctions


loss_func = WassersteinDistance() # this will eventually be modularized / removed

def _initialize_potential_param(net):
    potential_param = list(net.parameters())[-1].data
    potential_param = torch.zeros(potential_param.shape)
            
def _configure_psi(func):
    if hasattr(func, "psi_mu"):
        return func.psi_mu
    else:
        _initialize_potential_param(func)
        return func

class BaseModel(LightningModule):
    def __init__(self,
                 func,
                 train_t,
                 test_t,
                 optimizer=torch.optim.RMSprop,
                 dt=0.1,
                 burn_in_steps=100,
                 t_scale=0.02,
                 lr=1e-3,
                 seed=0):
        
        """
        Parameters:
        -----------
        func
            torch.nn.Module
        
        t_scale
            This is only used for odeint. Otherwise, it is None.
        """
        
        super(BaseModel, self).__init__()
                
            
        self._tau = 1e-6
        self._seed = seed
        torch.manual_seed(self._seed)
        self.hparam_dict = {}
        self.hparam_dict["seed"] = self._seed
        self.func = func
        self._t_scale = t_scale
        
        if not (hasattr(func, "mu") and hasattr(func, "sigma")):
            self.hparam_dict['t_scale'] = self._t_scale
        
        self.ForwardFunc = ForwardFunctions(self.func, self._t_scale)
            
        self._lr = lr
        self._train_t = train_t
        self._test_t = test_t
        self._dt = dt
        self._test_loss_list = []
        
        self.psi = _configure_psi(self.func)
        
        self._optimizer = optimizer
        
        
        for key, value in count_params(self).items():
            self.hparam_dict[key]=value
            
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            self.hparam_dict['device'] = gpu_name
        else:
            self.hparam_dict['device'] = 'cpu'
        
        
    def forward(self, x0, t, dt, tspan):

        """
        Parameters:
        -----------
        x0
            initial state

        t
            time

        To-Do:
        ------
        (1) Add other forward-stepping functions: odeint, prescient.forward_step()
        """

        return self.ForwardFunc.step(self.func,
                                     x0,
                                     t,
                                     dt,
                                     stdev=self._alpha,
                                     tspan=tspan,
                                     device=self.device,
                                    )

    def _fit_regularizer(self, x):

        x_final = self._X_final["X"].to(self.device)
        n_final = self._X_final["n_cells"]
        xf = x[:, -1, :]
        size_factor = n_final / xf.shape[0]

        burn_t_span = self._burn_t_final - self._X_final["t"]
        burn_dt = dt = burn_t_span / self._burn_in_steps
        burn_t = torch.Tensor([self._X_final["t"], self._burn_t_final])
        
        X_burn = self.forward(xf, burn_t, burn_dt, burn_t_span)
        burn_psi = size_factor * self.psi(X_burn).sum()
        final_psi = -1 * self.psi(x_final).sum()

        reg_psi = (final_psi + burn_psi) * self._tau

        return reg_psi

    def training_step(self, x):

        """
        To-do:
        ------
        (1) Currently the callback_metrics are very brittle. This must be generalized.

        Notes:
        ------
        (1) Required method of the pytorch_lightning.LightningModule subclass.
        """

        x0 = x[:, 0, :]
        x_obs = _restack_x(x, self._train_t)
        x_hat = self.forward(x0, self._train_t, self._dt, self._tspan['train'])
        xy_loss = loss_func.compute(x_hat, x_obs, self._train_t)
        if xy_loss.shape[0] > 1:
            self.log("train_loss_d4", xy_loss.detach()[0])
            self.log("train_loss_d6", xy_loss.detach()[1])
        else:
            self.log("train_loss_d6", xy_loss.detach()[0])
            
        if self._regularize:
            reg_loss = self._fit_regularizer(x)
            total_loss = reg_loss + xy_loss.sum()
            return total_loss
        else:
            return xy_loss.sum()

    def validation_step(self, x, idx):

        """ 
        
        Notes:
        ------
        
        
        To-do:
        ------
        (1) Currently the callback_metrics are very brittle. This must be generalized.
        """
                
        torch.set_grad_enabled(True)
        x0 = x[:, 0, :]
        x_obs = _restack_x(x, self._train_t)
        x_hat = self.forward(x0, self._train_t, self._dt, self._tspan['train'])
        xy_loss = loss_func.compute(x_hat, x_obs, self._train_t)
        if xy_loss.shape[0] > 1:
            self.log("val_loss_d4", xy_loss.detach()[0])
            self.log("val_loss_d6", xy_loss.detach()[1])
        else:
            self.log("val_loss_d6", xy_loss.detach()[0])
            
        return xy_loss.sum()

    def test_step(self, x, idx):

        """
        
        To-Do:
        ------
        (1) While this works, it is mostly implemented as a place-holder. We will need
            to update this.
        (2) Much of the code across the 3 {train, validation, test} steps are repeated.
            We need to make a more general forward function for these. 
        """
        
        torch.set_grad_enabled(True)
        x0 = x[:, 0, :]
        x_obs = _restack_x(x, self._test_t)
        x_hat = self.forward(x0, self._test_t, self._dt, self._tspan['test'])
        xy_loss = loss_func.compute(x_hat, x_obs, self._test_t)
        self.log("test_loss", xy_loss.detach())

        return xy_loss

    def configure_optimizers(self):
        """

        To-Do:
        ------
        (1) LR-scheduler (and multiple LR-scheduler configuration)

        Notes:
        ------
        (1) Required method of the pytorch_lightning.LightningModule subclass.
        """
        
        optimizer = self._optimizer(self.parameters(), lr=self._lr)
        return optimizer