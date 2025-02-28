# -- import packages: ---------------------------------------------------------
import abc
import ABCParse
import adata_query
import anndata
import autodevice
import lightning
import logging
import numpy as np
import torch

# -- import local dependencies: -----------------------------------------------
from .utils import L2Norm

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -- set type hints: ----------------------------------------------------------
from typing import Any, Dict, Optional, Union

# -- base class: --------------------------------------------------------------
class InstantaneousVelocity(ABCParse.ABCParse):
    """Quantify the instantaneous drift/diffusion given a state and model."""

    def __init__(self, silent: bool = False, *args, **kwargs) -> None:
        """Inhereting class should pass the model and device here."""

        self.__parse__(locals())
        self._L2Norm = L2Norm()

    @property
    def _t(self) -> None:
        return None

    @property
    def _DEVICE(self) -> torch.device:
        if isinstance(self._device, str):
            self._device = autodevice.AutoDevice(self._device)
        return self._device

    @property
    def _INPUT_IS_SIMULATED(self) -> bool:
        return "simulated" in self._adata.uns

    @property
    def _USE_KEY(self) -> str:
        """Some added flexibility for adata_sim, which places the target in adata.X"""
        if self._INPUT_IS_SIMULATED:
            try:
                self._use_key = adata_query.locate(self._adata_sim, self._use_key)
            except:
                return "X"
        return self._use_key

    @property
    def _X(self) -> torch.Tensor:
        return adata_query.fetch(
            self._adata, key=self._USE_KEY, torch=True, device=self._DEVICE
        )

    @property
    def _DiffEq(self):
        return self._model.DiffEq.DiffEq.to(self._DEVICE)

    def _format_outputs(self, X_pred: torch.Tensor) -> np.ndarray:
        """Reshape the output matrix and format as numpy array"""
        X_pred = X_pred.squeeze(dim=-1)
        if self._device.type == "cpu":
            return X_pred.detach().numpy()
        return X_pred.detach().cpu().numpy()

    @abc.abstractmethod
    def forward(self, X): ...

    def _key_action_indicator(self, key, keyset) -> str:
        if not key in keyset:
            return "Added"
        return "Updated"

    def _issue_info_message(self, key: str, key_action: str) -> None:
        logger.info(f"{key_action}: adata.obsm['{key}']")            

    def _add_to_adata(self, X_pred: np.ndarray) -> None:
        if not self._silent:
            logger.info(f"{key_action}: adata.obsm['{key}']")

    def _add_to_adata(self, X_pred: np.ndarray) -> None:
        """
        Add the predicted X_{term} to adata.obsm and compute the L2Norm(X_{term}),
        which is added to adata.obs[key].
        """

        key_action = self._key_action_indicator(
            key=self._obsm_key_added, keyset=self._adata.obsm_keys()
        )
        self._adata.obsm[self._obsm_key_added] = X_pred
        self._issue_info_message(key=self._obsm_key_added, key_action=key_action)

        key_action = self._key_action_indicator(
            key=self._obs_key_added, keyset=self._adata.obs_keys()
        )
        self._adata.obs[self._obs_key_added] = self._L2Norm(X_pred)
        self._issue_info_message(key=self._obs_key_added, key_action=key_action)

    def __call__(
        self,
        adata: anndata.AnnData,
        use_key: str,
        inplace: bool = True,
        *args,
        **kwargs,
    ):

        self.__update__(locals())

        X_pred = self._format_outputs(self.forward(self._X))  # np.ndarray

        if self._inplace:
            self._add_to_adata(X_pred)
        else:
            return X_pred


# -- controller class: --------------------------------------------------------
class InstantaneousDrift(InstantaneousVelocity):
    def __init__(
        self,
        model,
        obsm_key_added: str = "X_drift",
        obs_key_added: str = "drift",
        device=autodevice.AutoDevice(),
        silent: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """ """
        super().__init__(silent=silent)

        self.__parse__(locals())

    def forward(self, X: torch.Tensor):
        return self._DiffEq.f(self._t, X)


class InstantaneousDiffusion(InstantaneousVelocity):
    def __init__(
        self,
        model,
        obsm_key_added: str = "X_diffusion",
        obs_key_added: str = "diffusion",
        device=autodevice.AutoDevice(),
        silent: bool = False,
        *args,
        **kwargs,
    ) -> None:
        
        """ """
        
        super().__init__(silent=silent)

        self.__parse__(locals())

    def forward(self, X: torch.Tensor):
        return self._DiffEq.g(self._t, X)


# -- API-facing functions: ----------------------------------------------------
def drift(
    adata: anndata.AnnData,
    model: "sdq.scDiffEq",
    use_key: str = "X_pca",
    obsm_key_added: str = "X_drift",
    obs_key_added: str = "drift",
    device=autodevice.AutoDevice(),
    inplace: bool = True,
    silent: bool = False,
    *args,
    **kwargs,
):
    """Accessed as sdq.tl.drift(adata) or model.drift(adata)"""

    drift = InstantaneousDrift(
        model=model,
        obsm_key_added=obsm_key_added,
        obs_key_added=obs_key_added,
        device=device,
        silent=silent,
    )
    return drift(adata, use_key=use_key, inplace=inplace)


def diffusion(
    adata: anndata.AnnData,
    model: "sdq.scDiffEq",
    use_key: str = "X_pca",
    obsm_key_added: str = "X_diffusion",
    obs_key_added: str = "diffusion",
    device=autodevice.AutoDevice(),
    inplace: bool = True,
    silent: bool = False,
    *args,
    **kwargs,
):
    """Accessed as sdq.tl.diffusion(adata) or model.diffusion(adata)"""

    diffusion = InstantaneousDiffusion(
        model=model,
        obsm_key_added=obsm_key_added,
        obs_key_added=obs_key_added,
        device=device,
        silent=silent,
    )
    return diffusion(adata, use_key=use_key, inplace=inplace)
