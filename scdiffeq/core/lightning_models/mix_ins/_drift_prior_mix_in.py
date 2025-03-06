
class DriftPriorMixIn(object):
    logqp = True
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, Z0, t, **kwargs):
        """Forward step: (0) integrate in latent space"""
        
        return self.integrate(
            Z0=Z0,
            t=t,
            dt=self.hparams["dt"],
            logqp=self.logqp,
            **kwargs,
        )
            
    def log_computed_loss(self, sinkhorn_loss, t, kl_div_loss, stage):
        
        sinkhorn_loss = self.log_sinkhorn_divergence(
            sinkhorn_loss=sinkhorn_loss,
            t=t,
            stage=stage,
        ).sum()
        self.log(f"kl_div_{stage}", kl_div_loss.sum())

        return sinkhorn_loss + kl_div_loss.sum()

    def step(self, batch, batch_idx, stage=None):

        batch = self.process_batch(batch, batch_idx)
        X_hat, kl_div_loss = self.forward(batch.X0, batch.t)
        sinkhorn_loss = self.compute_sinkhorn_divergence(
            batch.X, X_hat, batch.W, batch.W_hat
        )
        return self.log_computed_loss(
            sinkhorn_loss, t=batch.t, kl_div_loss=kl_div_loss, stage=stage
        )

class DriftPriorVAEMixIn(object):
    logqp = True
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, X0, t, logqp=True, **kwargs):
        """Forward step: (0) integrate in latent space"""
        
        Z_hat, kl_div_loss = self.integrate(
            Z0=self.Encoder(X0),
            t=t,
            dt=self.hparams["dt"],
            logqp=logqp,
            **kwargs,
        )
        
        return self.Decoder(Z_hat), kl_div_loss
    
    def log_computed_loss(self, sinkhorn_loss, t, kl_div_loss, stage):
        
        sinkhorn_loss = self.log_sinkhorn_divergence(
            sinkhorn_loss=sinkhorn_loss,
            t=t,
            stage=stage,
        ).sum()
        self.log(f"kl_div_{stage}", kl_div_loss.sum())

        return sinkhorn_loss + kl_div_loss.sum()

    def step(self, batch, batch_idx, stage=None) -> None:
        
        train_optim = self.optimizers()[1]
        train_optim.zero_grad()
        
        batch = self.process_batch(batch, batch_idx)
        X_hat, kl_div_loss = self.forward(batch.X0, batch.t)
        sinkhorn_loss = self.compute_sinkhorn_divergence(
            batch.X, X_hat, batch.W, batch.W_hat
        )
        loss = self.log_computed_loss(
            sinkhorn_loss, t=batch.t, kl_div_loss=kl_div_loss, stage=stage
        )
        self.manual_backward(loss)
        train_optim.step()
