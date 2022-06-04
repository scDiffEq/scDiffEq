
__module_name__ = "_Learner.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# import packages #
# --------------- #
import torch


# import local dependencies #
# ------------------------- #
# from .. import _loss_functions as loss_funcs
from ._pass_to_model import _batched_no_grad_model_pass
from ._pass_to_model import _batched_training_model_pass

class _Learner:
    def __init__(
        self,
        model,
        lr,
        batch_size,
        device,
    ):
        self._device = device
        self._batch_size = batch_size
        self._Model = model
        self._optim_func = self._Model['optim']
        self._optimizer = self._optim_func(params=self._Model['params'], lr=lr)
        
                
        self._LossTracker = {"training":[], "validation":[]}
        self._training_epoch_count = 0
        self._training_loss = []

    def pass_train(self, X, t): #  X, t
        
        print("running learner.train()")


        self._optimizer.zero_grad()
        loss = _batched_training_model_pass(X,
                                            self._Model,
                                            self._optimizer,
                                            t,
                                            self._Model["reconst_loss_func"],
                                            self._Model["reparam_loss_func"],
                                            self._batch_size,
                                            self._device,
                                           )
        self._training_loss.append(loss)
        self._training_epoch_count += 1

    def pass_validation(self, X, t):

        loss = _batched_no_grad_model_pass(
            Model, X, t, reconst_loss_func, reparam_loss_func, device
        )

        self._validation_loss.append(loss)

    def pass_evaluation(self, X, t):

        self._test_loss = _batched_no_grad_model_pass(
            X, model, t, VAE, reconst_loss_func, reparam_loss_func, device
        )