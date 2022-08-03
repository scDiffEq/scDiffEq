
__module_name__ = "_BaseModel.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
from pytorch_lightning import LightningModule
import torchsde # will be removed eventually
import torch


# import local dependencies #
# ------------------------- #
from ._ancilliary._WassersteinDistance import WassersteinDistance
from ._ancilliary._shape_tools import _restack_x

loss_func = WassersteinDistance() # this will eventually be modularized / removed

class BaseModel(LightningModule):
    def __init__(self, func, train_t, test_t, dt=0.5, lr=1e-3):
        
        """
        Parameters:
        -----------
        func
            torch.nn.Module
        
        """
        
        super(BaseModel, self).__init__()

        self.func = func
        self._lr = lr
        self._train_t = train_t
        self._test_t = test_t
        self._dt = dt
        self._test_loss_list = []
        
    def forward(self, x, t):

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

        x0 = x[:, 0, :]
        x_hat = torchsde.sdeint(self.func, x0, t, dt=self._dt)
        x_obs = _restack_x(x, t)

        return x_hat, x_obs

    def training_step(self, x):

        """
        To-do:
        ------
        (1) Currently the callback_metrics are very brittle. This must be generalized.

        Notes:
        ------
        (1) Required method of the pytorch_lightning.LightningModule subclass.
        """

        x_hat, x_obs = self.forward(x, self._train_t)
        xy_loss = loss_func.compute(x_hat, x_obs, self._train_t)
        if xy_loss.shape[0] > 1:
            self.log("train_loss_4", xy_loss.detach()[0])
            self.log("train_loss_6", xy_loss.detach()[1])
        else:
            self.log("train_loss_6", xy_loss.detach()[0])
            
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
        x_hat, x_obs = self.forward(x, self._train_t)
        xy_loss = loss_func.compute(x_hat, x_obs, self._train_t)
        if xy_loss.shape[0] > 1:
            self.log("val_loss_4", xy_loss.detach()[0])
            self.log("val_loss_6", xy_loss.detach()[1])
        else:
            self.log("val_loss_6", xy_loss.detach()[0])
            
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
        self.x_hat, self.x_obs = self.forward(x, self._test_t)
        xy_loss = loss_func.compute(self.x_hat, self.x_obs, self._test_t)
        self.log("test_loss", xy_loss.detach())

        return xy_loss

    def configure_optimizers(self):
        """

        To-Do:
        ------
        (1) Multiple optimizer configuration
        (2) LR-scheduler (and multiple LR-scheduler configuration)

        Notes:
        ------
        (1) Required method of the pytorch_lightning.LightningModule subclass.
        """
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self._lr)
        return optimizer