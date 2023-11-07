
class BaseForwardMixIn(object):
    def __init__(self):
        super().__init__()


    def forward(self, Z0, t, **kwargs):
        """Forward step: (0) integrate in latent space"""
        return self.integrate(Z0=Z0, t=t, dt=self.hparams["dt"], logqp=False, **kwargs)
    
    def step(self, batch, batch_idx, stage=None):

        batch = self.process_batch(batch, batch_idx)
#         print(batch_idx, batch.X0.mean())
        X_hat = self.forward(batch.X0, batch.t)
        
#         print("X:", batch.X.shape)
#         print("X_hat:", X_hat.shape)
#         print("W (shape, first five):", batch.W.shape, batch.W[2][:5])
#         print("W_hat (shape, first five):", batch.W_hat.shape, batch.W_hat[2][:5])
        
        self.sinkhorn_loss = self.compute_sinkhorn_divergence(
            X = batch.X, X_hat = X_hat, W = batch.W, W_hat = batch.W_hat
        )
#         print("batch_loss", self.sinkhorn_loss.sum())
        return self.sinkhorn_loss.sum()
