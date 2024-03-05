
# -- import packages: ----------------------------------------------------------
import anndata
import ABCParse
import adata_query
import pandas as pd


# -- import local dependencies: ------------------------------------------------
from ..core import utils


# -- set typing: ---------------------------------------------------------------
from typing import Union, List, Dict


# -- controller class: ---------------------------------------------------------
class GroupedExpression(ABCParse.ABCParse):
    def __init__(
        self,
        adata: anndata.AnnData,
        gene_id_key: str = "gene_ids",
        use_key: str = "X_scaled",
        *args,
        **kwargs,
    ):
        """

        Parameters
        ----------
        adata: anndata.AnnData

        gene_id_key: str
            "gene_ids"

        use_key: str
            "X_scaled"

        Returns
        -------
        None

        Notes
        -----
        1.  If simulated AnnData is passed, you must have gene_id_key in adata.uns,
            i.e.,: in adata.uns[gene_id_key].
        """

        self.__parse__(locals(), ignore=["adata"], public=[None])
        self._process_adata(adata)

    @property
    def _SIMULATED_DETECTED(self):
        return self._gene_id_key in self.adata.uns

    def _process_simulated_anndata(self):
        """
        Updates self.adata and self._use_key
        """
        
        X_gene = adata_query.fetch(self.adata, key=self._use_key, torch=False)
        
        if isinstance(X_gene, pd.DataFrame):
            X_gene = X_gene.values

        var_names = self.adata.uns[self._gene_id_key]
        self.adata = anndata.AnnData(
            X=X_gene,
            dtype=X_gene.dtype,
            var=var_names.to_frame(),
            obs=self.adata.obs,
        )
        self._use_key = "X"

    def _process_adata(self, adata):

        self.adata = adata.copy()

        if self._SIMULATED_DETECTED:
            self._process_simulated_anndata()

        self.adata.var_names = self.adata.var[self._gene_id_key]

    def _single_gene_expression(self, df):
        return adata_query.fetch(
            adata = self.adata[df.index, self._gene_id],
            key=self._use_key,
            torch=False,
        )

    @property
    def _GROUPED(self):
        return self.adata.obs.groupby(self._groupby)

    def _process_output(self):
        return {
            self._gene_id: self._GROUPED.apply(self._single_gene_expression).to_dict()
        }

    def __call__(self, gene_id: str, groupby: str) -> Dict:
        """

        Parameters
        ----------
        gene_id: str

        groupby: str

        Returns
        -------
        grouped_expression: Dict

        Notes
        -----
        """

        self.__update__(locals(), private=["gene_id", "groupby"])

        return self._process_output()


# -- API-facing function: ------------------------------------------------------
def grouped_expression(
    adata: anndata.AnnData,
    gene_id: Union[List[str], str],
    groupby: str,
    gene_id_key: str = "gene_ids",
    use_key: str = "X_scaled",
    *args,
    **kwargs,
) -> Dict:

    """
    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object.

    gene_id: Union[List[str], str]
        Gene or list of genes for which to obtain grouped expression values.

    groupby: str
        obs_key by which to group cells.

    gene_id_key: str
        var_key to set var_names for fetching gene_ids.

    use_key: str
        Key to grab a specific matrix from AnnData.

    Returns
    -------
    grouped_expression: Dict

        Organized as:
        {
            gene_1: {
                group1: np.ndarray(...),
                group2: np.ndarray(...),
                ...
            },
            gene_2: {
                group1: np.ndarray(...),
                group2: np.ndarray(...),
                ...
            },
        }

    Notes
    -----
    1.  If simulated AnnData is passed, you must have gene_id_key in adata.uns,
        i.e.,: in adata.uns[gene_id_key].
    """

    if isinstance(gene_id, str):
        gene_id = [gene_id]

    grouped_expr = GroupedExpression(adata, gene_id_key=gene_id_key, use_key=use_key)
    return {_id: grouped_expr(_id, groupby)[_id] for _id in gene_id}
