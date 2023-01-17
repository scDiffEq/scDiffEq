
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
        seed=0,
        stdev: torch.nn.Parameter = None,
        expand: bool = False,
        use_key="X_pca",
        time_key="Time point",
        dt = 0.1,
        tau=1e-06,
        velo_gene_idx=None,
        V_coefficient = 0.2,
        V_scaling = 1,
        fate_scale=0,
        disable_velocity=False,
        disable_potential=False,
        disable_fate_bias=False,
        skip_positional_backprop=False,
        skip_positional_velocity_backprop=False,
        skip_potential_backprop=False,
        skip_fate_bias_backprop=False,
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
        self.V_scaling = V_scaling
        self.expand = expand
        
        
        self.save_hyperparameters(ignore=["adata", "self", "func", "stdev", "V_scaling"])
        self.hparams['func_description'] = str(func)
        self.save_hyperparameters(self.hparams)

    def forward(self, batch, batch_idx, stage=None):

        model_pass = {}
        forward_manager = ForwardManager(
            model=self,
            burn_steps=self.burn_steps,
            fate_scale=self.hparams['fate_scale'],
            velo_gene_idx=self.hparams['velo_gene_idx'],
            tau=self.hparams['tau'],
        )
        forward_outs = forward_manager(batch,
                                       batch_idx=batch_idx,
                                       stdev=self.stdev,
                                      ) #  t=t, 
        
        # need to clean this up -> pass model / parmas and then call from within as needed.
        loss_manager = LossManager(
            real_time=self.real_time,
            disable_velocity=self.hparams['disable_velocity'],
            disable_potential=self.hparams['disable_potential'],
            disable_fate_bias=self.hparams['disable_fate_bias'],
            skip_positional_backprop=self.hparams['skip_positional_backprop'],
            skip_positional_velocity_backprop=self.hparams['skip_positional_velocity_backprop'],
            skip_potential_backprop=self.hparams['skip_potential_backprop'],
            skip_fate_bias_backprop=self.hparams['skip_fate_bias_backprop'],
            tau = self.hparams['tau'],
            fate_scale = self.hparams['fate_scale'],
            V_coefficient = self.hparams['V_coefficient'], # provided
            V_scaling = self.V_scaling,     # learned
            model = self,
            stage = stage,
        )
        
        model_pass['batch'] = forward_manager.batch
        model_pass['X_hat'] = forward_outs['X_hat']
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
        return self.forward(batch, batch_idx, stage="train")

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
        return self.forward(self, batch, batch_idx, stage="val")

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
        return self.forward(batch, batch_idx, stage="test")

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
        return self.forward(batch, batch_idx, stage="predict") # , expand=self.expand) TODO: fix

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
