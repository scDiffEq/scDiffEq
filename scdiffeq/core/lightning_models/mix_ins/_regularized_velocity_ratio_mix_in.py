
import torch

from ....tools.utils import L2Norm

class RegularizedVelocityRatioMixIn(object):
    def __init__(self):
        """ """
        super().__init__()
        
        self._L2Norm = L2Norm()


    def forward(self, Z0, t, **kwargs):
        """Forward step: (0) integrate in latent space"""
        return self.integrate(Z0=Z0, t=t, dt=self.hparams["dt"], logqp=False, **kwargs)
    
    
    def _compute_magnitude_f(self, Z_hat: torch.Tensor) -> torch.Tensor:
        return self._L2Norm(self.DiffEq.drift(Z_hat)).mean(1) * self.hparams['dt']
        
    def _compute_magnitude_g(self, Z_hat: torch.Tensor) -> torch.Tensor:
        return self._L2Norm(torch.stack([self.DiffEq.diffusion(Z) for Z in Z_hat]).squeeze(-1)).mean(1)
    
    def _velocity_ratio_loss(self, Z_hat: torch.Tensor) -> torch.Tensor:
        """ """
        
        TARGET_RATIO = self.hparams['velocity_ratio_target']
        VELO_RATIO_ENFORCE = self.hparams['velocity_ratio_enforce']
        
        self.reg_f = self._compute_magnitude_f(Z_hat)
        self.reg_g = self._compute_magnitude_g(Z_hat)
        
        VELO_RATIO_LOSS = VELO_RATIO_ENFORCE * (TARGET_RATIO - self.reg_f.div(self.reg_g))
        return torch.abs(VELO_RATIO_LOSS)
    
    def step(self, batch, batch_idx, stage=None):

        batch = self.process_batch(batch, batch_idx)
        X_hat = self.forward(batch.X0, batch.t)
        self.sinkhorn_loss = self.compute_sinkhorn_divergence(
            X = batch.X, X_hat = X_hat, W = batch.W, W_hat = batch.W_hat
        )
        
        self.velocity_ratio_loss = self._velocity_ratio_loss(X_hat)
        
        if self.hparams['disable_velocity_ratio_backprop']:
            self.total_loss = self.sinkhorn_loss
        else:
            self.total_loss = self.sinkhorn_loss + self.velocity_ratio_loss

        return self.total_loss.sum()
