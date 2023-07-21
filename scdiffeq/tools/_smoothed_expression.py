# -- import packages: ----------------------------------------------------------
import anndata
import pandas as pd
<<<<<<< HEAD
import ABCParse

=======
>>>>>>> 0c2526d (add necessary funcs for smoothing gex)

# -- import local dependencies: ------------------------------------------------
from ..core import utils
from ._grouped_expression import GroupedExpression


# -- set typing: ---------------------------------------------------------------
from typing import Union, List, Dict, Optional


# -- controller class: ---------------------------------------------------------
<<<<<<< HEAD
class SmoothedExpression(ABCParse.ABCParse):
=======
class SmoothedExpression(utils.ABCParse):
>>>>>>> 0c2526d (add necessary funcs for smoothing gex)
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
        grouped_vals = pd.DataFrame(
            self._GROUPED_EXPRESSION(gene_id, groupby=self._time_key)[gene_id]
        )
        mean, std = grouped_vals.mean(0), grouped_vals.std(0)
        return {gene_id: pd.DataFrame({"mean": mean, "std": std})}

    @property
    def _GENE_IDS(self):
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
        gene_id: Union[List[str], str],
        return_dict: bool = False,
        *args,
        **kwargs,
    ):

        self.__update__(locals(), private=["adata_sim", "gene_id", "return_dict"])

        self._Results = {}
        for gene in self._GENE_IDS:
            self._Results.update(self.forward(gene))

        self._add_to_anndata()

        if self._return_dict:
            return self._Results


# -- API-facing function: ------------------------------------------------------
def smoothed_expression(
    adata_sim: anndata.AnnData,
    gene_id: Union[List[str], str],
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
    if return_dict:
        return result
