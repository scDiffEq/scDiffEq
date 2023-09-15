# -- import packages: ----------------------------------------------------------
import anndata
import pandas as pd
import ABCParse
import tqdm.notebook


# -- import local dependencies: ------------------------------------------------
from ..core import utils
from ._grouped_expression import GroupedExpression


# -- set typing: ---------------------------------------------------------------
from typing import Union, List, Dict, Optional

from typing import List


class SmoothedExpressionSummary(ABCParse.ABCParse):
    def __init__(
        self,
        keys: List[str] = ["mean", "std"],
        storage_key: str = "t_smoothed_gex",
        *args,
        **kwargs,
    ):

        self.__parse__(locals(), public=[None])
        self._INFO = utils.InfoMessage()

    def _isolate_per_gene(self, key: str):
        """
        key:
            "mean" or "std"
        """
        return pd.DataFrame(
            {
                gene: values[key].tolist()
                for gene, values in self._adata_sim.uns[self._storage_key].items()
            }
        )

    def forward(self):

        return {
            f"X_smoothed_gex_{key}": self._isolate_per_gene(key) for key in self._keys
        }

    def _add_to_adata(self, outputs):

        for key, val in outputs.items():
            description = key.split("_")[-1]
            self._adata_sim.uns[key] = val
            self._INFO(
                f"Smoothed {description} expression added to: `adata_sim.uns['{key}']`"
            )

    def __call__(self, adata_sim, *args, **kwargs):

        self.__update__(locals(), public=[None])

        self._add_to_adata(self.forward())


def summarize_smoothed_expression(
    adata_sim: anndata.AnnData,
    keys: List[str] = ["mean", "std"],
    storage_key: str = "t_smoothed_gex",
    *args,
    **kwargs,
) -> None:

    """
    Parameters
    ----------
    adata_sim: anndata.AnnData

    Returns
    -------
    None
    """

    smoothed_expression_summary = SmoothedExpressionSummary(
        keys=keys, storage_key=storage_key
    )
    smoothed_expression_summary(adata_sim)
    
# -- controller class: ---------------------------------------------------------
class SmoothedExpression(ABCParse.ABCParse):
    def __init__(
        self,
        time_key: str = "t",
        gene_id_key: str = "gene_ids",
        use_key: str = "X_gene",
        *args,
        **kwargs,
    ):

        self.__parse__(locals(), public=[None])

    @property
    def _GROUPED_EXPRESSION(self):
        if not hasattr(self, "_grouped_expr"):
            self._grouped_expr = GroupedExpression(
                adata=self._adata_sim,
                gene_id_key=self._gene_id_key,
                use_key=self._use_key,
            )
        return self._grouped_expr

    def _to_frame(self):
        ...

    def forward(self, gene_id: str):
        grouped_vals = self._GROUPED_EXPRESSION(gene_id, groupby=self._time_key)[gene_id]
        grouped_vals = pd.DataFrame({k: v.flatten() for k, v in grouped_vals.items()})
        mean, std = grouped_vals.mean(0), grouped_vals.std(0)
        return {gene_id: pd.DataFrame({"mean": mean, "std": std})}

    @property
    def _GENE_IDS(self):
        
        if self._gene_id is None:
            return self._adata_sim.uns[self._gene_id_key].tolist()
        
        if isinstance(self._gene_id, str):
            return [self._gene_id]
        
        return self._gene_id

    def _add_to_anndata(self):

        uns_key = f"{self._time_key}_smoothed_gex"

        if not uns_key in self._adata_sim.uns:
            self._adata_sim.uns[uns_key] = {}

        self._adata_sim.uns[uns_key].update(self._Results)

    def __call__(
        self,
        adata_sim: anndata.AnnData,
        gene_id: Optional[Union[List[str], str]] = None,
        return_dict: bool = False,
        *args,
        **kwargs,
    ):

        self.__update__(locals(), public = [None], ignore = ['gene_id'])
        self._gene_id = gene_id
        
        self._Results = {}
        for gene in tqdm.notebook.tqdm(self._GENE_IDS):
            self._Results.update(self.forward(gene))

        self._add_to_anndata()

        if self._return_dict:
            return self._Results


# -- API-facing function: ------------------------------------------------------
def smoothed_expression(
    adata_sim: anndata.AnnData,
    gene_id: Optional[Union[List[str], str]] = None,
    time_key: str = "t",
    gene_id_key: str = "gene_ids",
    use_key: str = "X_gene",
    return_dict: bool = False,
    *args,
    **kwargs,
):

    smoothed_expression = SmoothedExpression(
        time_key=time_key, gene_id_key=gene_id_key, use_key=use_key
    )
    result = smoothed_expression(adata_sim = adata_sim, gene_id=gene_id, return_dict=return_dict)
    
    summarize_smoothed_expression(
        adata_sim = adata_sim,
        storage_key = f"{time_key}_smoothed_gex",
        keys = ["mean", "std"],
    )

    if return_dict:
        return result
