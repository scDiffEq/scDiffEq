# -- import packages: ---------------------------------------------------------
import anndata
import autodevice
import logging
import torch

# -- import local dependencies: -----------------------------------------------
from ... import tools

# -- set type hints: ----------------------------------------------------------
from typing import Callable, Optional

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)


# -- mix-in cls: --------------------------------------------------------------
class StateCharacterizationMixIn(object):
    """MixIn container for state characterization functions."""

    def __check_adata(self, adata: anndata.AnnData) -> anndata.AnnData:
        """Check if adata was passed."""
        if adata is None:
            return self.adata
        return adata

    def __check_use_key(self, use_key: str) -> str:
        """Check if adata was passed."""
        if use_key is None:
            return self._use_key
        return use_key

    def __check_vital_inputs(self, adata: anndata.AnnData, use_key: str):
        """Check both adata and use_key"""
        adata = self.__check_adata(adata)
        use_key = self.__check_use_key(use_key)

        return adata, use_key

    def _drift_diffusion_fwd(
        self,
        func: Callable,
        obsm_key_added: str,
        obs_key_added: str,
        adata: Optional[anndata.AnnData] = None,
        use_key: Optional[str] = None,
        device=autodevice.AutoDevice(),
        inplace: bool = True,
        silent: bool = False,
        *args,
        **kwargs,
    ):
        """"""
        adata, use_key = self.__check_vital_inputs(adata, use_key)

        return func(
            adata=adata,
            model=self,
            use_key=use_key,
            obsm_key_added=obsm_key_added,
            obs_key_added=obs_key_added,
            device=device,
            inplace=inplace,
            silent=silent,
            *args,
            **kwargs,
        )

    def drift(
        self,
        adata: Optional[anndata._core.anndata.AnnData] = None,
        use_key: Optional[str] = None,
        obsm_key_added: str = "X_drift",
        obs_key_added: str = "drift",
        device=autodevice.AutoDevice(),
        inplace: bool = True,
        silent: bool = False,
        *args,
        **kwargs,
    ):
        """Drift method."""
        self._drift_diffusion_fwd(
            func=tools.drift,
            obsm_key_added=obsm_key_added,
            obs_key_added=obs_key_added,
            adata=adata,
            use_key=use_key,
            device=device,
            inplace=inplace,
            silent=silent,
            *args,
            **kwargs,
        )

    def diffusion(
        self,
        adata: Optional[anndata._core.anndata.AnnData] = None,
        use_key: Optional[str] = None,
        obsm_key_added: str = "X_diffusion",
        obs_key_added: str = "diffusion",
        device=autodevice.AutoDevice(),
        inplace: bool = True,
        silent: bool = False,
        *args,
        **kwargs,
    ):
        """Diffusion method."""

        self._drift_diffusion_fwd(
            func=tools.diffusion,
            obsm_key_added=obsm_key_added,
            obs_key_added=obs_key_added,
            adata=adata,
            use_key=use_key,
            device=device,
            inplace=inplace,
            silent=silent,
            *args,
            **kwargs,
        )

    def potential(
        self,
        use_key: str = "X_pca",
        raw_key_added: str = "_psi",
        norm_key_added: str = "psi",
        seed: int = 0,
        normalize: bool = True,
        return_raw_array: bool = False,
        q: float = 0.05,
        knn_smoothing_iters: int = 5,
        device: Optional[torch.device] = autodevice.AutoDevice(),
        use_tqdm: bool = True,
    ):
        """Potential method."""

        tools.cell_potential(
            adata=self.adata,
            model=self,
            use_key=use_key,
            raw_key_added=raw_key_added,
            norm_key_added=norm_key_added,
            device=device,
            seed=seed,
            normalize=normalize,
            return_raw_array=return_raw_array,
            q=q,
            knn_smoothing_iters=knn_smoothing_iters,
            use_tqdm=use_tqdm,
        )
