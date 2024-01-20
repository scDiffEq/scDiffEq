
# -- import packages: ---------------------------------------------------------
import abc
import numpy as np
import torch


# -- set typing: --------------------------------------------------------------
from typing import Union


# -- Base class: --------------------------------------------------------------
class FlexibleCall(abc.ABC):
    def __init__(self, *args, **kwargs):
        """"""

    @property
    def dtype(self):
        return type(self.input)

    @property
    def flexible_forward(self):
        if isinstance(self.input, np.ndarray):
            return self._numpy
        if isinstance(self.input, torch.Tensor):
            return self._torch

    @abc.abstractmethod
    def _numpy(self, input: np.ndarray, *args, **kwargs):
        ...

    @abc.abstractmethod
    def _torch(self, input: torch.Tensor, *args, **kwargs):
        ...

    def __call__(
        self, input: Union[np.ndarray, torch.Tensor], *args, **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        self.input = input
        return self.flexible_forward(self.input, *args, **kwargs)


# -- API-facing callables: ----------------------------------------------------
class L1Norm(FlexibleCall):
    """Manhattan Distance"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def _numpy(self, input: np.ndarray, axis=-1, **kwargs):
        return np.linalg.norm(input, ord=1, axis=axis, **kwargs)

    def _torch(self, input: torch.Tensor, dim=-1, **kwargs):
        return input.norm(p=1, dim=dim, **kwargs)

    def __repr__(self) -> str:
        return "L1 Norm: Manhattan Distance"


class L2Norm(FlexibleCall):
    """Euclidean Distance"""

    def __init__(self):
        super().__init__()

    def _numpy(self, input: np.ndarray, axis=-1, **kwargs):
        return np.linalg.norm(input, ord=2, axis=axis, **kwargs)

    def _torch(self, input: torch.Tensor, dim=-1):
        return input.norm(p=2, dim=dim)

    def __repr__(self) -> str:
        return "L2 Norm: Euclidean Distance"
