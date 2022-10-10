
__module_name__ = "_BaseModel.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages -------------------------------------------------------------
from pytorch_lightning import LightningModule, loggers
import torch


# import local dependencies ---------------------------------------------------
from . import _base_ancilliary as base
from . import _lightning_callbacks as cbs


# BaseModel: ------------------------------------------------------------------
class BaseModel(LightningModule):
    def __init__(self,
                 func=None,
                 log_path="./",
                 learning_rates=[1e-3],
                 optimizers=[torch.optim.Adam],
                 schedulers=[torch.optim.lr_scheduler.StepLR],
                 scheduler_kwargs=[{"step_size": 100, "gamma": 0.9}],
                 logger=loggers.CSVLogger,
                 flush_logs_every_n_steps=1,
                 logger_kwargs={},
                 dt=0.1,
                 alpha=0.5,
                 time_scale=None,
                 loss_function=base.SinkhornDivergence(),
                 trainer_kwargs={},
                 seed=617,
                 use_velocity=False, # TO-DO
                 use_fate_bias=True,
                 n_fates=None,
                 fate_bias_loss_function=torch.nn.functional.mse_loss,
                 fate_bias_loss_multiplier = 1,
                 callback_list=[cbs.SaveHyperParamsYAML(), cbs.TrainingSummary()],
                ):
        super(BaseModel, self).__init__()
        
        self.func = func
        self.save_hyperparameters(ignore=["func"])
        self.forward_integrate = base.ForwardFunctions(
            self.func,
            self.hparams['time_scale'],
            self.hparams['alpha'],
            self.hparams['dt'],
            self.device,
        )
        self._logger = logger(
            save_dir=self.hparams['log_path'],
            flush_logs_every_n_steps=self.hparams['flush_logs_every_n_steps'],
            **logger_kwargs,
        )
        self.t = torch.Tensor([2, 4, 6])
        
    def forward(self, X, batch_idx, t, stage):
        
        
        #### TO-DO: Update how we interpret incoming batch ----
               
        X0, X_obs, W_obs = base.format_batched_inputs(X, t)
        # equive to torchsde.sdeint
        X_hat = self.forward_integrate(self.func, X0=X0, t=t) # to-do: **kwargs 
        
        W_hat = torch.ones_like(W_obs, requires_grad=True).to(self.device)
        sinkhorn_loss = self.loss_function.compute(W_hat, X_hat, W_obs, X_obs, t)
        # to-do: make loss function usability more flexible to other LFs
        
        ### ------------------------------------------------------
        # fate bias loss
        # velocity loss
        # any other loss function
        # option to pass a function that is called during forward
        ### ------------------------------------------------------
        
        fwd_dict = {"X_hat": X_hat, "sinkhorn_loss": sinkhorn_loss,  "t": t}        
        fwd_dict['total_loss'] = sum(fwd_dict["sinkhorn_loss"])
        
#         base.update_loss_logs(
#             self,
#             fwd_dict["sinkhorn_loss"],
#             fwd_dict["t"],
#             self.current_epoch,
#             batch_idx,
#             stage,
#             "sinkhorn",
#             "d",
#             fate_bias_loss=self.hparams['use_fate_bias'],
#             fate_bias_metric="MSE",
#             fate_bias_multiplier=self.hparams['fate_bias_loss_multiplier'],
#         )
                
        return fwd_dict

    def training_step(self, batch, batch_idx):
        
        FWD = self.forward(X=batch, batch_idx=batch_idx, t=self.t, stage="val") # batch[-1].unique()
        return FWD['total_loss'] # requirement: return backprop

    def validation_step(self, batch, batch_idx):

        base.retain_gradients(self)
        FWD = self.forward(X=batch, batch_idx=batch_idx,  t=self.t, stage="val")
        return FWD['total_loss']

    def test_step(self, batch, batch_idx):

        base.retain_gradients(self)
        FWD = self.forward(X=batch, batch_idx=batch_idx,  t=self.t, stage="test")
        return FWD['total_loss']

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        FWD = self.forward(X=batch, batch_idx=batch_idx,  t=self.t, stage="predict")
        return FWD
    
    def configure_optimizers(self):
        return base.optim_config(
            param_groups=[self.parameters()],
            learning_rates=self.hparams['learning_rates'],
            optimizers=self.hparams['optimizers'],
            schedulers=self.hparams['schedulers'],
            scheduler_kwargs=self.hparams['scheduler_kwargs'],
        )