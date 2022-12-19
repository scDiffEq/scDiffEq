
import torch

from ..utils import sum_normalize
from ._sinkhorn_divergence import SinkhornDivergence


NoneType = type(None)
from typing import Union

class Loss:
    def __parse__(self, kwargs, ignore=["self"], explicit=["X", "X_hat"]):

        for key, val in kwargs.items():
            if not key in ignore:
                if not key in explicit:
                    key = "_{}".format(key)
                setattr(self, key, val)

    def __init__(self):
        """Catch the outputs of the inference function"""
        self.__parse__(locals())
        self.sinkhorn_divergence = SinkhornDivergence()
        self.mse = torch.nn.MSELoss()

    @property
    def W(self):
        if not isinstance(self._W, torch.Tensor):
            return sum_normalize(torch.ones_like(self.X)[:, :, 0])
        return self._W

    @property
    def W_hat(self):
        if not isinstance(self._W_hat, torch.Tensor):
            return sum_normalize(torch.ones_like(self.X_hat)[:, :, 0])
        return self._W_hat

    @property
    def V(self):
        if not isinstance(self._V, torch.Tensor):
            return sum_normalize(torch.ones_like(self.W))
        return self._V

    @property
    def V_hat(self):
        if not isinstance(self._V_hat, torch.Tensor):
            return sum_normalize(torch.ones_like(self.W_hat))
        return self._V_hat

    @property
    def F(self):
        if not isinstance(self._F, torch.Tensor):
            return sum_normalize(torch.ones_like(self.W))
        return self._F

    @property
    def F_hat(self):
        if not isinstance(self._F_hat, torch.Tensor):
            return sum_normalize(torch.ones_like(self.W_hat))
        return self._F_hat

    def positional(self):
        return self.sinkhorn_divergence(self.W, self.X, self.W_hat, self.X_hat)

    def velocity(self):
        return self.mse(self.V, self.V_hat)

    def fate(self):
        return self.mse(self.F, self.F_hat)

    def __call__(
        self,
        X: Union[torch.Tensor],
        X_hat: Union[torch.Tensor],
        W: Union[torch.Tensor, NoneType] = None,
        W_hat: Union[torch.Tensor, NoneType] = None,
        V: Union[torch.Tensor, NoneType] = None,
        V_hat: Union[torch.Tensor, NoneType] = None,
        F: Union[torch.Tensor, NoneType] = None,
        F_hat: Union[torch.Tensor, NoneType] = None,
    ):
        self.__parse__(locals())

        loss_dict = {}
        # -- always do this
        loss_dict["positional"] = self.positional()
        loss_dict["velocity"] = self.velocity()
        loss_dict["fate"] = self.fate()

        return loss_dict