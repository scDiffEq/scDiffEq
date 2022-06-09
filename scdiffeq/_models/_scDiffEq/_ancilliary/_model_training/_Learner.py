
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

    def pass_train(self, X, t, pretrain_VAE): #  X, t

        self._optimizer.zero_grad()        
        self._X_pred, self._loss = _batched_training_model_pass(X,
                                            self._Model,
                                            self._optimizer,
                                            t,
                                            self._Model["reconst_loss_func"],
                                            self._Model["reparam_loss_func"],
                                            self._batch_size,
                                            pretrain_VAE,
                                            self._device,
                                           )
        self._loss = self._loss.mean(0)
        sum_loss = self._loss.sum().item()
        print("| Training Loss: | d4: {:.3f} d6: {:.3f} | Total: {:.3f}".format(self._loss[0].item(),
                                                                 self._loss[1].item(),
                                                                 sum_loss))
        self._LossTracker["training"].append(self._loss)
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