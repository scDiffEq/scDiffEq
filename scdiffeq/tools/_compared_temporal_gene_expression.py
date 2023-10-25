from typing import Optional, Union, List, Dict
import numpy as np
import sklearn
import anndata
import ABCParse

from ..core import utils
from ._annotate_gene_features import annotate_gene_features
from ._grouped_expression import grouped_expression


class TemporalGeneExpression(ABCParse.ABCParse):
    def __init__(
        self,
        adata: anndata.AnnData,
        adata_sim: anndata.AnnData,
        gene_id_key="gene_ids",
        sim_time_key="t",
        ref_time_key="Time point",
        sim_use_key="X_gene",
        ref_use_key="X_scaled",
        PCA: Optional[sklearn.decomposition.PCA] = None,
    ):
        self.__parse__(locals(), ignore=["adata", "adata_sim", "PCA"], public=[None])

        self.adata_sim = adata_sim.copy()
        self.adata = adata.copy()
        self.adata.var_names = adata.var[self._gene_id_key]

        annotate_gene_features(self.adata_sim, self.adata, PCA=PCA)

    @property
    def REF_GEX(self):
        return grouped_expression(
            self.adata,
            gene_id=self._gene_id,
            groupby=self._ref_time_key,
            use_key=self._ref_use_key,
        )

    @property
    def SIM_GEX(self):
        return grouped_expression(
            self.adata_sim,
            gene_id=self._gene_id,
            groupby=self._sim_time_key,
            use_key=self._sim_use_key,
        )

    def __call__(self, gene_id: Union[List[str], str], *args, **kwargs) -> Dict:

        self.__update__(locals(), private=["gene_id"])

        return {
            "ref": self.REF_GEX,
            "sim": self.SIM_GEX,
        }


def compared_temporal_gene_expression(
    adata: anndata.AnnData,
    adata_sim: anndata.AnnData,
    gene_id: Union[List[str], str],
    gene_id_key: str = "gene_ids",
    sim_time_key: str = "t",
    ref_time_key: str = "Time point",
    sim_use_key: str = "X_gene",
    ref_use_key: str = "X_scaled",
    PCA: Optional[sklearn.decomposition.PCA] = None,
    *args,
    **kwargs
) -> Dict:
    
    """
    Compare gene expression across pseudo-time-steps and real time.
    
    Parameters
    ----------
    adata: anndata.AnnData,
    
    adata_sim: anndata.AnnData,
    
    gene_id: Union[List[str], str],
    
    gene_id_key: str = "gene_ids",
    
    sim_time_key: str = "t",
    
    ref_time_key: str = "Time point",
    
    sim_use_key: str = "X_gene",
    
    ref_use_key: str = "X_scaled",
    
    PCA: Optional[sklearn.decomposition.PCA] = None,
    
    Returns
    -------
    compared_gex: Dict
    
    Notes
    -----
    
    """
    time_gex = TemporalGeneExpression(
        adata=adata,
        adata_sim=adata_sim,
        gene_id_key=gene_id_key,
        sim_time_key=sim_time_key,
        ref_time_key=ref_time_key,
        sim_use_key=sim_use_key,
        ref_use_key=ref_use_key,
        PCA=PCA,
    )
    return time_gex(gene_id=gene_id)
