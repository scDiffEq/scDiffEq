
# -- import packages: ---------------------------------------------------------
import ABCParse
import anndata
import numpy as np
import pandas as pd


# -- operational class: -------------------------------------------------------
class PseudotimeBinning(ABCParse.ABCParse):
    def __init__(self, n_bins: int = 10, *args, **kwargs):
        """

        Args:
            n_bins (int): number of non-t0 bins. **Default**: 10.

        """
        self.__parse__(locals())

        self._n_bins = self._n_bins + 2

    @property
    def _BOUNDS(self):
        if not hasattr(self, "_bounds"):
            self._bounds = np.linspace(0, 1, self._n_bins)
        return self._bounds

    @property
    def bins(self):
        if not hasattr(self, "_bins"):
            self._bins = np.array(
                [[i, j] for i, j in zip(self._BOUNDS[:-1], self._BOUNDS[1:])]
            )
        return self._bins

    def _assign_bin(self, value, bins):
        """"""
        match_idx = np.all([value >= self.bins[:, 0], value <= self.bins[:, 1]], axis=0)
        assigned_bin = self.bins[match_idx].flatten()

        return pd.Series(
            {
                "ti": assigned_bin[0],
                "tj": assigned_bin[1],
                "bin": np.where(match_idx)[0][0],
            }
        )

    def _integrate_binned_time(
        self, adata: anndata.AnnData, time_df: pd.DataFrame
    ) -> None:
        """ """
        adata.obs = pd.merge(adata.obs, time_df, how="left", on="index")
        adata.obs["bin"] = adata.obs["bin"].astype(int)
        adata.obs = adata.obs.rename({"bin": "t"}, axis=1)

    @property
    def _TIME_BIN_COLS_PRESENT(self) -> bool:
        return any([key in self._adata.obs for key in ["ti", "tj", "bin", "t"]])

    def __call__(self, adata: anndata.AnnData, pseudotime_key: str):
        """ """

        self.__update__(locals())

        if not self._TIME_BIN_COLS_PRESENT:
            self._time_df = self._adata.obs[self._pseudotime_key].apply(
                self._assign_bin, bins=self.bins
            )
            self._integrate_binned_time(self._adata, self._time_df)


# -- API-facing function: -----------------------------------------------------
def bin_pseudotime(adata: anndata.AnnData, pseudotime_key: str, n_bins: int = 10):
    """
    Args:
        adata (anndata.AnnData)

        pseudotime_key (str) (e.g., "ct_pseudotime")

        n_bins (int) Default = 10

    Returns:
        None
    """
    time_binning = PseudotimeBinning(n_bins=n_bins)
    time_binning(adata=adata, pseudotime_key=pseudotime_key)
