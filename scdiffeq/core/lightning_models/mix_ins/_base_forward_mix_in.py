
class BaseForwardMixIn(object):
    def __init__(self):
        super().__init__()


    def forward(self, Z0, t, **kwargs):
        """Forward step: (0) integrate in latent space"""
        return self.integrate(Z0=Z0, t=t, dt=self.hparams["dt"], logqp=False, **kwargs)

    def step(self, batch, batch_idx, stage=None):

        batch = self.process_batch(batch, batch_idx)
        X_hat = self.forward(batch.X0, batch.t)
        sinkhorn_loss = self.compute_sinkhorn_divergence(
            batch.X, X_hat, batch.W, batch.W_hat
        )
        return self.log_sinkhorn_divergence(sinkhorn_loss, t=batch.t, stage=stage)