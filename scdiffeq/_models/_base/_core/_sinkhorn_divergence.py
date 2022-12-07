
__module_name__ = "_sinkhorn_divergence.py"
__doc__ = """To-do."""
__author__ = ", ".join(["Michael E. Vinyard", "Anders Rasmussen", "Ruitong Li"])
__email__ = ", ".join(
    [
        "mvinyard@broadinstitute.org",
        "arasmuss@broadinstitute.org",
        "ruitong@broadinstitute.org",
    ]
)


# import packages: -----------------------------------------------------------------------
from abc import ABC, abstractmethod
from geomloss import SamplesLoss
import inspect
import torch


# import local dependencies: -------------------------------------------------------------
from .._utils import (
#     autodevice,
    extract_func_kwargs,
    local_arg_parser,
)


# supporting functions: ------------------------------------------------------------------
def _sum_norm(W, sample_axis=1):
    return W / W.sum(sample_axis)[:, None]


def _format_W(W, X, sample_axis=1):
    if not isinstance(W, torch.Tensor):
        W = torch.ones_like(X[:, :, 0, None])
    if any(W.sum(sample_axis) > 1):
        return _sum_norm(W, sample_axis)
    return W


# Base class: ----------------------------------------------------------------------------
class LossFunction(ABC):
#     device = autodevice()

    def __parse__(self, passed_kwargs):

        setattr(self, "sample_axis", passed_kwargs["sample_axis"])
        passed_kwargs = local_arg_parser(passed_kwargs)
        func_kwargs = extract_func_kwargs(self.loss_func, passed_kwargs)
        self.loss_func = self.loss_func(**func_kwargs)

    @abstractmethod
    def __call__(self):
        pass


# API-facing class: ----------------------------------------------------------------------
class SinkhornDivergence(LossFunction):
    loss_func = SamplesLoss

    def __init__(
        self,
        device,
        loss="sinkhorn",
        backend="online",
        p=2,
        blur=0.1,
        scaling=0.7,
        debias=True,
        sample_axis=1,
        **kwargs
    ):
        self.device = device
        self.__parse__(locals())

    def __format_inputs__(
        self,
        X: torch.Tensor,
        X_hat: torch.Tensor,
        W: torch.Tensor = None,
        W_hat: torch.Tensor = None,
    ):
        """
        Prepare inputs for Sinkhorn call.

        Parameters:
        -----------
        X
            type: torch.Tensor

        X_hat
            type: torch.Tensor

        W
            type: torch.Tensor
            default: None

        W_hat
            type: torch.Tensor
            default: None

        Returns:
        --------
        formatted_kwargs

        Notes:
        ------
        (1) Assumes v is organized with 't' as dim=0
        (2) If W or W_hat is missing, it is filled in using a uniform, sum-normalized
            torch.ones Tensor. Even if done elsewhere, doesn't impact values.
        """
        passed_kwargs = locals()
        passed_kwargs.pop("self")

        passed_kwargs["W"] = _format_W(
            passed_kwargs["W"], passed_kwargs["X"], self.sample_axis
        )
        passed_kwargs["W_hat"] = _format_W(
            passed_kwargs["W_hat"], passed_kwargs["X_hat"], self.sample_axis
        )
        self.passed_kwargs = passed_kwargs

        formatted_kwargs = {}
        for k, v in passed_kwargs.items():
            formatted_kwargs[k] = v[1:].float().to(self.device).requires_grad_()

        return formatted_kwargs

    def __call__(
        self,
        X: torch.Tensor,
        X_hat: torch.Tensor,
        W: torch.Tensor = None,
        W_hat: torch.Tensor = None,
    ):
        """
        X
            type: torch.Tensor
        X_hat
            type: torch.Tensor
        W
            default: None
        W_hat
            default: None

        Notes:
        ------
        """
        
        inputs = self.__format_inputs__(X, X_hat, W, W_hat)
            
        return self.loss_func(
            inputs["W"], inputs["X"], inputs["W_hat"], inputs["X_hat"]
        )
