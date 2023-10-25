
import ABCParse
import anndata
import numpy as np
import pandas as pd

from typing import Optional, Union, Dict, List, Tuple

class TemporalExpression(ABCParse.ABCParse):
    def __init__(
        self,
        time_key: str = "t",
        groupby: Optional[Union[None, str]] = None,
        gene_id_key: str = "gene_ids",
        use_key: str = "X_gene_inv",
        *args,
        **kwargs,
    ):
        self.__parse__(locals(), public=[None])

    @property
    def _GROUPBY(self):
        """
        if no additional groupby key is passed, we just group by time.
        """
        if self._groupby is None:
            return self._time_key

        groupby = [self._time_key]
        groupby.append(self._groupby)
        return groupby

    @property
    def _GROUPED(self):
        if not hasattr(self, "_grouped"):
            self._grouped = self._adata_sim.obs.groupby(self._GROUPBY)
        return self._grouped

    @property
    def _GENE(self):
        if not hasattr(self, "_gene") or (self._gene is None):
            self._gene = self._adata_sim.uns[self._gene_id_key].tolist()
        return self._gene

    @property
    def _MULTIGROUPED(self):
        return isinstance(self._GROUPBY, List)

    def _forward(self, df, operation=np.mean):
        gene = self._adata_sim[df.index].obsm[self._use_key][self._GENE]
        return operation(gene)

    def _add_to_adata(self, mean, std):
        """"""
        self._adata_sim.uns["gex_mean"] = mean
        self._INFO("Mean temporal expression added to: `adata_sim.uns['gex_mean']`")
        self._adata_sim.uns["gex_std"] = std
        self._INFO(
            "Standard deviation of temporal expression added to: `adata_sim.uns['gex_std']`"
        )

    def __call__(
        self,
        adata_sim: anndata.AnnData,
        gene: Optional[Union[List[str], str]] = None,
        return_gex: bool = False,
        *args,
        **kwargs,
    ) -> Union[None, Tuple[pd.DataFrame]]:

        """
        Parameters
        ----------
        adata_sim: anndata.AnnData

        gene: Optional[Union[List[str], str]], default = None

        return_gex: bool, default = False

        Returns
        -------
        mean, std: Union[None, Tuple[pd.DataFrame]]
            If `return_gex` is True, returns values.
        """

        self.__update__(locals(), public=[None])

        mean = self._GROUPED.apply(self._forward, operation=np.mean)
        std = self._GROUPED.apply(self._forward, operation=np.std)

        if self._MULTIGROUPED:
            mean, std = mean.unstack(), std.unstack()

        self._add_to_adata(mean, std)

        if return_gex:
            return mean, std


def temporal_expression(
    adata_sim: anndata.AnnData,
    groupby: Optional[Union[None, str]] = "final_state",
    gene: Optional[Union[List[str], str]] = None,
    time_key: str = "t",
    gene_id_key: str = "gene_ids",
    use_key: str = "X_gene_inv",
    return_gex: bool = False,
) -> Union[None, Tuple[pd.DataFrame]]:
    """
    Parameters
    ----------
    adata_sim: anndata.AnnData

    groupby: Optional[Union[None, str]], default = "final_state"

    gene: Optional[Union[List[str], str]], default = None

    time_key: str, default = "t"

    gene_id_key: str, default = "gene_ids"

    use_key: str, default = "X_gene_inv"

    return_gex: bool, default = False

    Returns
    -------
    mean, std: Union[None, Tuple[pd.DataFrame]]
        If `return_gex` is True, returns values.
    """

    t_expr = TemporalExpression(
        time_key=time_key, groupby=groupby, gene_id_key=gene_id_key, use_key=use_key
    )
    return t_expr(adata_sim=adata_sim, gene=gene, return_gex=return_gex)
