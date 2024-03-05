import lightning
import torch

from typing import Any, Dict, Union

class ModelLogging(lightning.Callback):
    def __init__(self, *args, **kwargs):
        
        self._CURRENT_EPOCH = None
        self._EPOCH_TRAIN_LOSS = []
        self._EPOCH_VALIDATION_LOSS = []
            
    def on_train_epoch_end(self, trainer, pl_module):
        
        epoch_train_loss = torch.hstack(self._EPOCH_TRAIN_LOSS).mean().to(torch.float32)
        
        pl_module.log("epoch_train_loss", epoch_train_loss)
        self._EPOCH_TRAIN_LOSS = []
        
    def on_validation_epoch_end(self, trainer, pl_module):
        
        epoch_validation_loss = torch.hstack(self._EPOCH_VALIDATION_LOSS).mean().to(torch.float32)
        pl_module.log("epoch_validation_loss", epoch_validation_loss)
        self._EPOCH_VALIDATION_LOSS = []
        
    def log_total_epochs(self, pl_module):
        """Train model N times --> N"""
        pl_module.log("total_epochs", torch.Tensor([pl_module.COMPLETED_EPOCHS]).to(torch.float32))

    def log_lr(self, pl_module):

        if not isinstance(pl_module.optimizers(), list):
            lr = pl_module.optimizers().optimizer.state_dict()["param_groups"][0]["lr"]
            pl_module.log("opt_param_group_lr", lr)
        else:
            for i, opt in enumerate(pl_module.optimizers()):
                for j, pg in enumerate(opt.optimizer.state_dict()["param_groups"]):
                    pl_module.log(f"opt_{i}_param_group_{j}_lr", pg["lr"])

    def log_sinkhorn_divergence(self, pl_module, t, stage: str):
        
        if hasattr(pl_module, "sinkhorn_loss"):
            sinkhorn_loss = pl_module.sinkhorn_loss

            for i in range(len(t)):
                _t = round(t[i].item(), 3)
                pl_module.log(f"sinkhorn_{_t}_{stage}", sinkhorn_loss[i])

            return sinkhorn_loss.sum()
        
    def log_f(self, pl_module, t, stage: str):
        if hasattr(pl_module, "reg_f"):
            for i in range(len(t)):
                _t = round(t[i].item(), 3)
                pl_module.log(f"velo_f_{_t}_{stage}", pl_module.reg_f[i])

    def log_g(self, pl_module, t, stage: str):
        if hasattr(pl_module, "reg_g"):
            for i in range(len(t)):
                _t = round(t[i].item(), 3)
                pl_module.log(f"velo_g_{_t}_{stage}", pl_module.reg_g[i])

    def log_velocity_ratio_loss(self, pl_module, t, stage: str):
        
        if hasattr(pl_module, "velocity_ratio_loss"):
            velocity_ratio_loss = pl_module.velocity_ratio_loss

            for i in range(len(t)):
                _t = round(t[i].item(), 3)
                pl_module.log(f"velo_ratio_{_t}_{stage}", velocity_ratio_loss[i])

            return velocity_ratio_loss.sum()

    def on_train_batch_end(
        self, 
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Union[torch.Tensor, Dict[str, Any]],
        batch: Any,
        batch_idx: int,
    ):
                
        batch = pl_module.process_batch(batch, batch_idx)
        
        self.log_total_epochs(pl_module)
        self.log_lr(pl_module)
        sinkhorn_total = self.log_sinkhorn_divergence(
            pl_module = pl_module, t = batch.t, stage = "training",
        )
        self._EPOCH_TRAIN_LOSS.append(sinkhorn_total.detach().cpu())
        velocity_ratio_loss_total = self.log_velocity_ratio_loss(
            pl_module = pl_module, t = batch.t, stage = "training",
        )
        # right now, we do nothing with the VRL total
        self.log_f(pl_module = pl_module, t = batch.t, stage = "training")
        self.log_g(pl_module = pl_module, t = batch.t, stage = "training")

    def on_validation_batch_end(
        self, 
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Union[torch.Tensor, Dict[str, Any]],
        batch: Any,
        batch_idx: int,
    ):
        
        batch = pl_module.process_batch(batch, batch_idx)
        
        self.log_total_epochs(pl_module)
        self.log_lr(pl_module)
        sinkhorn_total = self.log_sinkhorn_divergence(
            pl_module = pl_module, t = batch.t, stage = "validation",
        )
        self._EPOCH_VALIDATION_LOSS.append(sinkhorn_total.detach().cpu())
        velocity_ratio_loss_total = self.log_velocity_ratio_loss(
            pl_module = pl_module, t = batch.t, stage = "validation",
        )
        # right now, we do nothing with the VRL total
        self.log_f(pl_module = pl_module, t = batch.t, stage = "validation")
        self.log_g(pl_module = pl_module, t = batch.t, stage = "validation")
