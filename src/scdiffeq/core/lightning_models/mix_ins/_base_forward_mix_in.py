
class BaseForwardMixIn(object):
    
    """
    Base class implementing forward mixing functionality.

    This mixin provides methods for forward mixing, which is a common operation
    in certain types of neural network models, particularly those related to
    sequence generation or transformation tasks.

    Attributes:
        None

    Methods:
        __init__: Initialize the BaseForwardMixIn object.
        forward: Perform the forward step, integrating in the latent space.
        step: Perform a single optimization step using the forward mixing approach.

    """
    
    def __init__(self) -> None:
        """
        Initialize the BaseForwardMixIn object.

        This initializes the object.

        Args:
            None

        Returns:
            None

        """
        super().__init__()


    def forward(self, Z0, t, **kwargs):
        """
        Perform the forward step: integrate in the latent space.

        This method integrates the given latent space Z0 over time t using
        specific parameters.

        Args:
            Z0 (tensor): Initial latent space.
            t (tensor): Time steps.
            **kwargs: Additional keyword arguments.

        Returns:
            tensor: Resulting integrated latent space.

        """
        return self.integrate(Z0=Z0, t=t, dt=self.hparams["dt"], logqp=False, **kwargs)
    
    def step(self, batch, batch_idx, stage=None):
        """Perform a single optimization step using the forward mixing approach.

        This method takes a batch of data and performs a single optimization step
        using the forward mixing approach. It calculates the Sinkhorn divergence loss
        between the actual and predicted outputs.

        Args:
            batch (Batch): Input batch of data.
            batch_idx (int): Index of the batch.
            stage (str): Optional stage information.

        Returns:
            tensor: Computed Sinkhorn divergence loss.

        """
        batch = self.process_batch(batch, batch_idx)
        X_hat = self.forward(batch.X0, batch.t)
        self.sinkhorn_loss = self.compute_sinkhorn_divergence(
            X = batch.X, X_hat = X_hat, W = batch.W, W_hat = batch.W_hat
        )
        return self.sinkhorn_loss.sum()
