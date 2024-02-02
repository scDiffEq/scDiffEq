
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
    
    @property
    def velocity_ratio_loss_transform(self):
        if not hasattr(self, "_velocity_ratio_loss_transform"):
            self._velocity_ratio_loss_transform = getattr(torch, self.hparams['velocity_ratio_params']['method'])
        return self._velocity_ratio_loss_transform
    
    def _compute_magnitude_f(self, Z_hat: torch.Tensor) -> torch.Tensor:
        return self._L2Norm(self.DiffEq.drift(Z_hat)).mean(1) * self.hparams['dt']
        
    def _compute_magnitude_g(self, Z_hat: torch.Tensor) -> torch.Tensor:
        return self._L2Norm(torch.stack([self.DiffEq.diffusion(Z) for Z in Z_hat]).squeeze(-1)).mean(1)

    def _velocity_ratio_loss(self, Z_hat: torch.Tensor) -> torch.Tensor:
        """ """
        
        vr_params = self.hparams['velocity_ratio_params']
        
        self.reg_f = self._compute_magnitude_f(Z_hat)
        self.reg_g = self._compute_magnitude_g(Z_hat)
        
        return vr_params["enforce"] * self.velocity_ratio_loss_transform(
            vr_params["target"] - self.reg_f.div(self.reg_g)
        )
    
    def _dynamic_enforcement(self):

        """
        From Anders: if you want to do the moving average, just grab the last N
        sinkhorn loss quantities from that list. and take the mean of that as your
        "velo_ratio_enforce." If its the first time computing the loss you can use
        velo_ratio_enforce = 0 instead of an average. this also means the VELO_RATIO_LOSS
        will change over training, decreasing as the model gets better at reconstructing
        the data.
        """
        
        # TODO
        
        ...
        
    def step(self, batch, batch_idx, stage=None):

        batch = self.process_batch(batch, batch_idx)
        X_hat = self.forward(batch.X0, batch.t)
        self.sinkhorn_loss = self.compute_sinkhorn_divergence(
            X = batch.X, X_hat = X_hat, W = batch.W, W_hat = batch.W_hat
        )
        
        self.velocity_ratio_loss = self._velocity_ratio_loss(X_hat)
        
        self.total_loss = self.sinkhorn_loss + self.velocity_ratio_loss

        return self.total_loss.sum()
