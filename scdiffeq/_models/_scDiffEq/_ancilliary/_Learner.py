
__module_name__ = "_Learner.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import torch


# import local dependencies #
# ------------------------- #
from ._loss_functions._OptimalTransportLoss import _OptimalTransportLoss as OTLoss
from ._loss_functions._KL_Divergence import _KL_Divergence as KLDivLoss

from ._model_functions._pass_to_model import _batched_no_grad_model_pass
from ._model_functions._pass_to_model import _batched_training_model_pass


class _Learner:
    def __init__(
        self,
        VAE,
        NeuralDiffEq,
        parameters,
        lr,
        device,
        Model_HyperParams,
        optim_func=torch.optim.RMSprop,
        reconstruction_loss_function=OTLoss,
        reparameterization_loss_function=KLDivLoss,
    ):

        self._VAE = VAE
        self._NeuralDiffEq = NeuralDiffEq
        self._parameters = parameters
        self._lr = lr
        self._device = device
        self._Model_HyperParams = Model_HyperParams
        self._optimizer = optim_func(self._parameters, lr=self._lr)
        self._reconst_loss_func = reconstruction_loss_function(self._device)
        self._reparam_loss_func = reparameterization_loss_function
        self._training_epoch_count = 0
        self._training_loss = []
        self._validation_loss = []

    def train(self, X, t):

        self._optimizer.zero_grad()
        loss = _batched_training_model_pass(
            X, model, optimizer, t, VAE, reconst_loss_func, reparam_loss_func, device
        )
        self._training_loss.append(loss)
        self._training_epoch_count += 1

    def validate(self, X, t):

        loss = _pass_to_model_no_grad(
            X, model, t, VAE, reconst_loss_func, reparam_loss_func, device
        )

        self._validation_loss.append(loss)

    def evaluate(self, X, t):

        self._test_loss = _pass_to_model_no_grad(
            X, model, t, VAE, reconst_loss_func, reparam_loss_func, device
        )