


# -- import packages: --------------------------------------------------------------------
import torch


# -- import local dependencies: ----------------------------------------------------------
from ...utils import sum_normalize, AutoParseBase
from ._sinkhorn_divergence import SinkhornDivergence


# -- import typing: ----------------------------------------------------------------------
NoneType = type(None)
from typing import Union


# -- main class: -------------------------------------------------------------------------
class Loss(AutoParseBase):

    def __init__(self):
        """Catch the outputs of the inference function"""
        self.__parse__(locals(), public=["X", "X_hat"])
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

#     @property
#     def V(self):
#         if not isinstance(self._V, torch.Tensor):
#             return sum_normalize(torch.ones_like(self.W))
#         return self._V

#     @property
#     def V_hat(self):
#         if not isinstance(self._V_hat, torch.Tensor):
#             return sum_normalize(torch.ones_like(self.W_hat))
#         return self._V_hat
    
#     @property
#     def V_confidence(self):
#         if not isinstance(self._V_confidence, torch.Tensor):
#             return sum_normalize(torch.ones_like(self.V_hat))
#         return self._V_confidence
    
#     @property
#     def V_hat_confidence(self):
#         if not isinstance(self._V_hat_confidence, torch.Tensor):
#             return sum_normalize(torch.ones_like(self.V_confidence))
#         return self._V_hat_confidence

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
#         print("W\tX\tW_hat\tX_hat")
#         print(self.W.shape, self.X.shape, self.W_hat.shape, self.X_hat.shape)
        return self.sinkhorn_divergence(self.W, self.X, self.W_hat, self.X_hat)

    @property
    def V_hat(self):
        return torch.diff(self.X_hat, n=1, dim=0, append=self.X_hat[-1:, :, :])
    
    def positional_velocity(self):
        """Meant to be temporary and deleted"""
        
        W = torch.concat([self.W, self.W], axis=1)
        print(self.X.shape, self._V.shape)
        XV = torch.concat([self.X, self._V], axis=1)
        W_hat = torch.concat([self.W_hat, self.W_hat], axis=1)
        XV_hat = torch.concat([self.X_hat, self.V_hat], axis=1)
        
        return self.sinkhorn_divergence(W, XV, W_hat, XV_hat)
        
#     def velocity(self):
#         return self.sinkhorn_divergence(self.V_confidence, self.V, self.V_hat_confidence, self.V_hat)

    def fate(self):
        return self.mse(self.F, self.F_hat)

    def __call__(
        self,
        X: Union[torch.Tensor],
        X_hat: Union[torch.Tensor],
        W: Union[torch.Tensor, NoneType] = None,
        W_hat: Union[torch.Tensor, NoneType] = None,
        V: Union[torch.Tensor, NoneType] = None,
#         V_hat: Union[torch.Tensor, NoneType] = None,
#         V_confidence: Union[torch.Tensor, NoneType] = None,
#         V_hat_confidence: Union[torch.Tensor, NoneType] = None,
        F: Union[torch.Tensor, NoneType] = None,
        F_hat: Union[torch.Tensor, NoneType] = None,
    ):
        self.__parse__(locals(), public=["X", "X_hat"])

        loss_dict = {}
        loss_dict["positional"] = self.positional()
        loss_dict['positional_velocity'] = self.positional_velocity()
#         loss_dict["velocity"] = self.velocity()
#         loss_dict["fate"] = self.fate()

        return loss_dict