
__module_name__ = "__init__.py"
__version__ = "0.0.45"
__doc__ = """TODO"""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# -- import packages: --------------------------------------------------------------------
from pytorch_lightning import LightningModule
from neural_diffeqs import NeuralODE, NeuralSDE
from torch_nets import TorchNet
from torchsde import sdeint
import torch


# -- local impoirts: ---------------------------------------------------------------------
from ._default_neural_sde import default_NeuralSDE
from ..utils import function_kwargs


from ..forward import ForwardManager, LossManager


NoneType = type(None)


# -- LightningModel: ---------------------------------------------------------------------
class LightningDiffEq(LightningModule):
    """Pytorch-Lightning model trained within scDiffEq"""
    
    def __parse__(self, kwargs, ignore=["self", "func", "__class__"]):
        
        self.kwargs = {}
        for key, val in kwargs.items():
            if not key in ignore:
                self.kwargs[key] = val
                if key == "kwargs":
                    for k, v in val.items():
                        self.kwargs[k] = v
                        
                        
    def enable_grad(self):
        if any([self.mu_is_potential, self.sigma_is_potential]):
            torch.set_grad_enabled(True)

    def __init__(
        self,
        func: [NeuralSDE, NeuralODE, TorchNet] = None,
        stdev: torch.nn.Parameter = None,
        expand: bool = False,
        dt = 0.1,
        tau=1e-06,
        **kwargs,
    ):
        """
        Parameters:
        -----------
        func
        
        lit_config
        
        kwargs
        
        Returns:
        --------
        None
        """

        super(LightningDiffEq, self).__init__()
        self.__parse__(locals())
        self.func = func
        self.stdev = stdev
        self.expand = expand
        
        
        self.save_hyperparameters(ignore=["adata", "self", "func", "stdev"])
        self.hparams['func_description'] = str(func)
        self.save_hyperparameters(self.hparams)

    def forward(self, batch, batch_idx, stage, stdev=0.5, t=None):

        model_pass = {}
        forward_manager = ForwardManager(
            model=self, tau=self.tau, burn_steps=self.burn_steps
        )
        forward_outs = forward_manager(batch, batch_idx=batch_idx, stdev=stdev) #  t=t, 
        loss_manager = LossManager(real_time=self.real_time)
        
        model_pass['batch'] = forward_manager.batch
        model_pass['X_hat'] = forward_outs['X_hat']

        forward_outs['fate_scale'] = self.fate_scale
        model_pass['loss'] = loss_manager(**forward_outs)
        
        return model_pass

    def training_step(self, batch, batch_idx)->dict:
        """
        Wraps "LightningModel.forward", indicating stage as "train".
        
        Parameters:
        -----------
        batch
            type: list
        
        batch_idx
            type: int
        
        Returns:
        --------
        forward_out
            Contains at least "loss" key, required for PyTorch-Lightning backprop.
            type: dict
        """
        return self.forward(batch, batch_idx, stage="train", stdev=0.5, t=None)

    def validation_step(self, batch, batch_idx):
        """
        Wraps "LightningModel.forward", indicating stage as "val".
        
        Parameters:
        -----------
        batch
            type: list
        
        batch_idx
            type: int
        
        Returns:
        --------
        forward_out
            Contains at least "loss" key, required for PyTorch-Lightning backprop.
            type: dict
        """        
        self.enable_grad()
        return self.forward(self, batch, batch_idx, stage="val", stdev=0.5, t=None)

    def test_step(self, batch, batch_idx):
        """
        Wraps "LightningModel.forward", indicating stage as "test".
        
        Parameters:
        -----------
        batch
            type: list
        
        batch_idx
            type: int
        
        Returns:
        --------
        forward_out
            Contains at least "loss" key, required for PyTorch-Lightning backprop.
            type: dict
        """
        self.enable_grad()
        return self.forward(self, batch, batch_idx, stage="test", stdev=0.5, t=None)

    def predict_step(self, batch, batch_idx):
        """
        Wraps "LightningModel.forward", indicating stage as "predict".
        
        Parameters:
        -----------
        batch
            type: list
        
        batch_idx
            type: int
        
        Returns:
        --------
        forward_out
            Contains at least "loss" key, required for PyTorch-Lightning backprop.
            type: dict
        """
        self.enable_grad()
        return self.forward(batch, batch_idx, stage="predict", stdev=0.5, t=self.t) # , expand=self.expand) TODO: fix

    def configure_optimizers(self):
        """
        Parameters:
        -----------
        None

        Returns:
        --------
        [optimizer], [scheduler]
            type: list, list

        TODO:
        -----
        (1) add documentation
        (2) add support for multiple optimizers & schedulers (needed or not?)
        """
        
        return [self.optimizer], [self.lr_scheduler]
