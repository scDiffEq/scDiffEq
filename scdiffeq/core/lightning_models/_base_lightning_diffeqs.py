
# -- import packages: ----------------------------------------------------------
from pytorch_lightning import LightningModule
from abc import ABC, abstractmethod
from autodevice import AutoDevice
import torch

NoneType = type(None)


# -- import local dependencies: ------------------------------------------------
from ..utils import AutoParseBase


# -- Parent Class: -------------------------------------------------------------
class BaseLightningDiffEq(LightningModule, AutoParseBase):
    """Pytorch-Lightning model trained within scDiffEq"""
    def __init__(self):
        super(BaseLightningDiffEq, self).__init__()
        
        self.save_hyperparameters(ignore=['func', 'pca'])
    
    @abstractmethod
    def process_batch(self):
        ...
        
    @abstractmethod
    def forward(self):
        ...
        
    @abstractmethod
    def loss(self):
        ...
        
    def record(self, loss, stage):
        """Record loss. called in step"""
        
        log_msg = "{}"
        if not isinstance(stage, NoneType):
            log_msg = f"{stage}_" + "{}"
        for i, l in enumerate(loss):
            self.log(log_msg.format(i), l.item())
            
    @abstractmethod
    def step(self, batch, batch_idx, stage=None):
        ...
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="training")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="validation")
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="test")
    
    def predict_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="predict")
    
    def configure_optimizers(self):
        return self.optimizers, self.lr_schedulers


# -- base sub-classes: ---------------------------------------------------------
class BaseLightningDriftNet(BaseLightningDiffEq):
    def __init__(self, dt=0.1, stdev=torch.Tensor([0.5])):
        super(BaseLightningDriftNet, self).__init__()
        
        from brownian_diffuser import nn_int
        self.nn_int = nn_int
            
    def forward(self, X0, t, stage=None, max_steps=None, return_all=False):
        return self.nn_int(
            self.func,
            X0=X0,
            t=t,
            dt=self.hparams["dt"],
            stdev=self.hparams["stdev"],
            max_steps=max_steps,
            return_all=return_all,
        )

class BaseLightningODE(BaseLightningDiffEq):
        
    def __init__(self):
        super(BaseLightningODE, self).__init__()
        from torchdiffeq import odeint
        self.odeint = odeint

    def forward(self, X0, t, stage=None, **kwargs):
        """
        We want this to be easily-accesible from the outside, so we
        directly define the forward step with the integrator code.
        """
        return self.odeint(self.func, X0, t=t, **kwargs)


class BaseLightningSDE(BaseLightningDiffEq):
        
    def __init__(self, dt=0.1):
        super(BaseLightningSDE, self).__init__()
        from torchsde import sdeint
        self.sdeint = sdeint

    def forward(self, X0, t, stage=None, **kwargs):
        """
        We want this to be easily-accesible from the outside, so we
        directly define the forward step with the integrator code.
        """
                
        return self.sdeint(self.func, X0, ts=t, dt=self.hparams["dt"], **kwargs)

class BaseVeloDiffEq(BaseLightningDiffEq):
    from sklearn.decomposition import PCA
    def __init__(self, pca=None, velo_gene_idx=None, gene_space_key="X_scaled", dt=0.1, n_pcs=50):
        super(BaseVeloDiffEq, self).__init__()
                
    @property
    def pca_model(self):
        if isinstance(self._pca_model, NoneType):            
            
            print("doing PCA...", end="")
            self._pca_model = self.PCA(n_components=self.hparams['n_pcs'])
            X_gene = self.adata.layers[self.hparams['gene_space_key']]
            self._pca_model.fit_transform(X_gene)
            print("done")
            
        return self._pca_model
        
    def inferred_velocity(self, X_hat):
        
        """
        Inferred instantaneous velocity at a given position from the delta 
        of the observed positions
        """

        X_hat_np = X_hat.detach().cpu().numpy()
        X_hat_ = torch.stack(
            [torch.Tensor(self.pca_model.inverse_transform(xt)) for xt in X_hat_np]
        ).to(AutoDevice())
        X_hat_ = X_hat_[:, :, self.hparams['velo_gene_idx']]
        
        return torch.diff(X_hat_, n=1, dim=0, append=X_hat_[-1][None, :, :])
    
    def loss(self, W, X, W_hat, X_hat, V, V_hat):
        
        state_loss = self.loss_func(W, X, W_hat, X_hat)
        X = torch.concat([X, V], axis=-1).contiguous()
        X_hat = torch.concat([X_hat, V_hat], axis=-1).contiguous()
        velo_loss = self.loss_func(W, X, W_hat, X_hat)

        return {"state": state_loss, "velo": velo_loss}
