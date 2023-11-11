
class BaseForwardMixIn(object):
    def __init__(self):
        super().__init__()


    def forward(self, Z0, t, **kwargs):
        """Forward step: (0) integrate in latent space"""
        return self.integrate(Z0=Z0, t=t, dt=self.hparams["dt"], logqp=False, **kwargs)
    
    def step(self, batch, batch_idx, stage=None):

        batch = self.process_batch(batch, batch_idx)
        X_hat = self.forward(batch.X0, batch.t)
        X = batch.X
        W = batch.W
        W_hat = batch.W_hat
        
        print(f"W: {W}")
        print(f"W_hat: {W_hat}")
        
        self.sinkhorn_loss = self.compute_sinkhorn_divergence(
            X = batch.X, X_hat = X_hat, W = batch.W, W_hat = batch.W_hat
        )
        return self.sinkhorn_loss.sum()
