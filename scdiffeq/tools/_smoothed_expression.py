# -- import packages: ----------------------------------------------------------
import anndata
import pandas as pd
import tqdm

# -- import local dependencies: ------------------------------------------------
from ..core import utils
from ._grouped_expression import GroupedExpression


# -- set typing: ---------------------------------------------------------------
from typing import Union, List, Dict, Optional


# -- controller class: ---------------------------------------------------------
class SmoothedExpression(utils.ABCParse):
    def __init__(
        self,
        time_key: str = "t",
        gene_id_key: str = "gene_ids",
        use_key: str = "X_gene",
        disable_tqdm: bool = False,
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
        
    @property
    def _GENE_ID_PROGRESS_BAR(self):
        if self._disable_tqdm:
            return self._GENE_IDS
        return tqdm.notebook.tqdm(self._GENE_IDS)

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
        for gene in self._GENE_ID_PROGRESS_BAR:
            self._Results.update(self.forward(gene))

        self._add_to_anndata()

        if self._return_dict:
            return self._Results


class SmoothedFrameVarmHandler:
    """
    Operating class to mediate adding smoothed expression matrices to adata.varm
    in the case that all genes are passed.
    """
    def __init__(self):
        ...

    def _to_frame(self, adata, key):
        return pd.DataFrame(
            {gene: expr[key] for gene, expr in adata.uns["t_smoothed_gex"].items()}
        )

    @property
    def mean(self):
        if not hasattr(self, "_mean"):
            self._mean = self._to_frame(self._adata, key="mean")
        return self._mean

    @property
    def std(self):
        if not hasattr(self, "_std"):
            self._std = self._to_frame(self._adata, key="std")
        return self._std

    def __call__(self, adata: anndata.AnnData, return_dfs: bool = False):

        self._adata = adata

        self._adata.varm["smoothed_mean"] = self.mean.T
        self._adata.varm["smoothed_std"] = self.std.T

        if return_dfs:
            return self.mean, self.std

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
    
    """
    Parameters
    ----------
    adata_sim: anndata.AnnData
        Simulated AnnData object.
        
    gene_id: Optional[Union[List[str], str]], default = None
        Gene name. If None, all genes are used. Called from adata_sim.var_names
        
        ...
    """
    
    if gene_id is None:
        gene_id = adata_sim.var_names.tolist()

    smoothed_expression = SmoothedExpression(
        time_key=time_key, gene_id_key=gene_id_key, use_key=use_key
    )
    result = smoothed_expression(adata_sim = adata_sim, gene_id=gene_id, return_dict=return_dict)
    
    if len(gene_id) == adata_sim.shape[1]:
        varm_handler = SmoothedFrameVarmHandler()
        varm_handler(adata_sim, return_dfs = False)
    
    if return_dict:
        return result
