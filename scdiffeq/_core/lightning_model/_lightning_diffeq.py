
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
from autodevice import AutoDevice
from torch_nets import TorchNet
from torchsde import sdeint
import torch


# -- local impoirts: ---------------------------------------------------------------------
from ._default_neural_sde import default_NeuralSDE
from ..utils import AutoParseBase, function_kwargs, LoggingLearnableHParams
from .forward import ForwardManager, LossManager


NoneType = type(None)


# -- LightningModel: ---------------------------------------------------------------------
class LightningDiffEq(LightningModule, AutoParseBase):
    """Pytorch-Lightning model trained within scDiffEq"""                        
                        
    def enable_grad(self):
        if any([self.mu_is_potential, self.sigma_is_potential]):
            torch.set_grad_enabled(True)

    def __init__(
        self,
        adata,
        func: [NeuralSDE, NeuralODE, TorchNet] = None,
        optimizers=["SGD","RMSprop",],
        lr_schedulers=["StepLR", "StepLR"],
        learning_rates = [1e-8, 1e-4],
        func_type = None,
        lr=1e-4,
        seed=0,
        stdev: [torch.nn.Parameter, torch.Tensor] = torch.Tensor([0.]),
        pretrain_stdev = torch.Tensor([0.5]).to(AutoDevice()),
        pretrain_lr = 1e-6,
        pretrain_tau = torch.Tensor([1]).to(AutoDevice()),
        pretrain_burn_steps=50,
        expand: bool = False,
        use_key="X_pca",
        time_key="Time point",
        t=None,
        dt = 0.1,
        real_time=True,
        adjoint=False,
        velo_gene_idx=None,
        V_coefficient = 0.2,
        V_scaling = torch.Tensor([1]),
        tau=torch.Tensor([1e-06]),
        fate_scale=0,
        burn_steps=100,
        mu_is_potential=False,
        sigma_is_potential=False,
        disable_velocity=False,
        disable_potential=False,
        disable_fate_bias=False,
        skip_positional_backprop=False,
        skip_positional_velocity_backprop=False,
        skip_potential_backprop=False,
        skip_fate_bias_backprop=False,
        automatic_optimization=False,
        lightning_ignore=["self"],
        base_ignore=["self"],
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
        
        self.__config__(locals())
        
    def __config__(self, kwargs):
                
        self.__parse__(kwargs, private=[None])

        self.save_hyperparameters(ignore=kwargs['lightning_ignore'])
        self.hparams['func_description'] = str(self.func)
        
        learnable_params = ['stdev', 'tau', 'V_scaling']
        for param in learnable_params:
            log_learnable_param = LoggingLearnableHParams(kwargs[param])
            self.hparams[param] = log_learnable_param()
            
        self.save_hyperparameters(self.hparams, ignore=kwargs['lightning_ignore'])
        self._pretrain = False
        
        for attr_name in ['stdev', 'lr', 'tau', 'burn_steps']:
            attr_name_ = "pretrain_{}".format(attr_name)
            attr = getattr(self, attr_name_)
            if isinstance(attr, type(None)):
                setattr(self, attr_name_, getattr(self, "{}".format(attr_name)))

        net_params = list(self.func.parameters())
        net_params[-1].data = torch.zeros(net_params[-1].data.shape) # initialize
    
    @property
    def pretrain(self):
        return self._pretrain
    
    @pretrain.setter
    def pretrain(self, val):
        self._pretrain = val

    #     def _forward_pretrain(self, batch, batch_idx, stage=None):
        
#         forward_manager = ForwardManager(model=self)
#         forward_outs = forward_manager(batch, batch_idx=batch_idx, stage=stage)
#         return forward_outs
    
    def forward(self, batch, batch_idx, stage=None):
        
        optimizers = {"pretrain": self.optimizers[0], "train": self.optimizers[1]}
                
        if self.pretrain:
            optimizers['pretrain'].zero_grad()
            forward_manager = ForwardManager(model=self)
            forward_outs = forward_manager(batch, batch_idx=batch_idx, stage=stage)
            loss = forward_outs
            self.manual_backward(loss)
            optimizers['pretrain'].step()

        else:
            optimizers['train'].zero_grad()
            forward_manager = ForwardManager(model=self)
            forward_outs = forward_manager(batch, batch_idx=batch_idx, stage=stage)
            loss_manager = LossManager(model=self, stage=stage)
            loss = {
                'batch': forward_manager.batch,
                'X_hat': forward_outs['X_hat'],
                'loss':  loss_manager(**forward_outs),
                     }
            self.manual_backward(loss['loss'])
            optimizers['train'].step()
        return loss        
        

    def training_step(self, batch, batch_idx): # , optimizer_idx)->dict:
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
#         print("training_step | optimizer: {}".format(optimizer_idx))
        
        return self.forward(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx): # optimizer_idx
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
        return self.forward(batch, batch_idx, stage="val")

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

#     def optimizer_step(
#         self,
#         epoch,
#         batch_idx,
#         optimizer,
#         optimizer_idx,
#         optimizer_closure,
#         second_order_closure=None,
#         using_native_amp=None,
#         on_tpu=False,
#         using_lbfgs=False,
#     ):
#         if (self.pretrain and (optimizer_idx == 0)):
#             optimizer.step(closure=optimizer_closure)
# #             print("opt: {}\t| pretrain: {}\t| backprop: True".format(optimizer_idx, self.pretrain))
#         elif optimizer_idx == 1:
#             optimizer.step(closure=optimizer_closure)
# #             print("opt: {}\t| pretrain: {}\t| backprop: True".format(optimizer_idx, self.pretrain))
#         else:
# #             print("opt: {}\t| pretrain: {}\t| backprop: False".format(optimizer_idx, self.pretrain))
#             optimizer_closure()

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

        return self.optimizers, self.lr_schedulers
