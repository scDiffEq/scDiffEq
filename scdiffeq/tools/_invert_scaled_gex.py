# -- import packages: ---------------------------------------------------------
import ABCParse
import adata_query
import anndata
import numpy as np
import pandas as pd
import sklearn.preprocessing


# -- set typing: --------------------------------------------------------------
from typing import Optional, Union, List


# -- Operational class: -------------------------------------------------------
class InvertScalingExpression(ABCParse.ABCParse):
    def __init__(self, scaler_model, *args, **kwargs):

        self.__parse__(locals(), public=[None])

    def _configure_X_gene(self):

        X_gene = adata_query.fetch(self._adata_sim, self._use_key)

        if isinstance(X_gene, pd.DataFrame):
            X_gene = X_gene.values

        self.ndim = X_gene.ndim
        if self.ndim == 2:
            return X_gene[None, :, :]
        elif self.ndim == 3:
            return X_gene

    @property
    def X_gene(self):
        if not hasattr(self, "_X_gene_processed"):
            self._X_gene_processed = self._configure_X_gene()
        return self._X_gene_processed

    def invert(self):
        """Assumes X_gene is a 3-D array of batch x t x gene"""
        return np.stack(
            [self._scaler_model.inverse_transform(batch) for batch in self.X_gene]
        )

    def adjust_negative_values(self):
        self.X_unscaled[np.where(self.X_unscaled < 0)] = 0
        return self.X_unscaled

    @property
    def _PD_KWARGS(self):

        pd_kwargs = {"index": self._adata_sim.obs.index}

        if not hasattr(self, "_gene_ids") and (
            "gene_ids" in self._adata_sim.uns_keys()
        ):
            setattr(self, "_gene_ids", self._adata_sim.uns["gene_ids"])
        if hasattr(self, "_gene_ids"):
            pd_kwargs.update({"columns": self._gene_ids})

        return pd_kwargs

    def _format_outputs(self, use_key, suffix_added):
        """
        gene_ids
        adata_sim
        """

        if self.ndim == 2:
            self.X_gene_inv = self.X_gene_inv.squeeze(0)

        self.X_gene_inv = pd.DataFrame(self.X_gene_inv, **self._PD_KWARGS)
        self._adata_sim.obsm[f"{use_key}_{suffix_added}"] = self.X_gene_inv

        return self.X_gene_inv

    def __call__(
        self,
        adata_sim,
        use_key: str = "X_gene",
        suffix_added: str = "inv",
        return_df: bool = False,
        gene_ids: Optional[Union[pd.Series, pd.Index, List]] = None,
        silent: bool = False,
        *args,
        **kwargs,
    ) -> pd.DataFrame:

        self.__update__(locals(), public=[None])

        self.X_unscaled = self.invert()
        self.X_gene_inv = self.adjust_negative_values()

        self.X_gene_inv = self._format_outputs(
            use_key=use_key, suffix_added=suffix_added
        )

        if return_df:
            return self.X_gene_inv


# -- API-facing function: -----------------------------------------------------
def invert_scaled_gex(
    adata_sim: anndata.AnnData,
    scaler_model: sklearn.preprocessing.StandardScaler,
    use_key: str = "X_gene",
    suffix_added: str = "inv",
    return_df: bool = False,
    gene_ids: Optional[Union[pd.Series, pd.Index, List]] = None,
    *args,
    **kwargs,
) -> Union[pd.DataFrame, None]:

    """
    Parameters
    ----------
    adata_sim: anndata.AnnData
        Simulated AnnData object. Required.

    scaler_model: sklearn.preprocessing.StandardScaler
        Scaler model used during preprocessing. Required.

    use_key: str, default = "X_gene"
        Key to access the scaled, predicted counts.

    suffix_added: str, default = "inv"

    return_df: bool, default = False
        True, if the pd.DataFrame should be returned in addition to updating
        AnnData. If False, only AnnData is Updated.

    gene_ids: Optional[Union[pd.Series, pd.Index, List]], default = None
        Gene values. Will be assigned as the columns of the resulting
        pd.DataFrame.

    Returns
    -------
    None or X_gene_inv: Union[pd.DataFrame, None]. Updates `adata_sim` either way.

    Notes
    -----
    For a more granular approach to inverting scaler gene
    expression, use `ReverseScalingExpression`.
    """

    inverter = InvertScalingExpression(scaler_model)
    X_gene_inv = inverter(
        adata_sim=adata_sim,
        use_key=use_key,
        suffix_added=suffix_added,
        return_df=return_df,
        gene_ids=gene_ids,
        *args,
        **kwargs,
    )

    if return_df:
        return X_gene_inv
