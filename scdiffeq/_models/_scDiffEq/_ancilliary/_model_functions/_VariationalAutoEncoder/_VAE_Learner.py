
__module_name__ = "_Learner.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import torch


def _pass_data_through_model(model, X):

    """encode, reparameterize, decode"""

    model.encode(X)
    model.reparameterize()
    model.decode()

    return model._X_reconstructed, model._mu, model._log_var


def _compute_total_loss(
    reconstruction_loss_function,
    reparameterization_loss_function,
    X_input,
    X_reconstructed,
    mu,
    log_var,
):

    """compute and sum reconstruction loss, reparameterization loss"""
    
    reconstruction_loss = reconstruction_loss_function(X_input, X_reconstructed)
    reparameterization_loss = reparameterization_loss_function(mu, log_var)
    total_loss = reconstruction_loss + reparameterization_loss
    
    return total_loss


def _run_VAE(model, X, reconstruction_loss_function, reparameterization_loss_function):

    """"""

    X_reconstructed, mu, log_var = _pass_data_through_model(model, X)
    total_loss = _compute_total_loss(
        reconstruction_loss_function=reconstruction_loss_function,
        reparameterization_loss_function=reparameterization_loss_function,
        X_input=X,
        X_reconstructed=X_reconstructed,
        mu=mu,
        log_var=log_var,
    )
    return total_loss

class _Learner:
    def __init__(
        self,
        model,
    ):

        self._model = model
        self._optimizer = self._model._optimizer
        self._reconstruction_loss_function = self._model._reconstruction_loss_function
        self._reparameterization_loss_function = (
            self._model._reparameterization_loss_function
        )
        self._training_epoch_count = 0
        self._training_loss = []
        self._validation_loss = []

    def train(self, X):

        self._optimizer.zero_grad()
        loss = _run_VAE(
            self._model,
            X,
            self._reconstruction_loss_function,
            self._reparameterization_loss_function,
        )
        loss.backward()
        self._optimizer.step()
        loss = loss.detach().cpu()
        self._training_loss.append(loss.item())
        self._training_epoch_count += 1

    def validate(self, X):
        with torch.no_grad():
            loss = _run_VAE(
                self._model,
                X,
                self._reconstruction_loss_function,
                self._reparameterization_loss_function,
            )
            loss = loss.detach().cpu()
            self._validation_loss.append(loss.item())

    def evaluate(self, X):
        with torch.no_grad():
            self._test_loss = _run_VAE(
                self._model,
                X,
                self._reconstruction_loss_function,
                self._reparameterization_loss_function,
            )
            self._test_loss = self._test_loss.detach().cpu().item()
            