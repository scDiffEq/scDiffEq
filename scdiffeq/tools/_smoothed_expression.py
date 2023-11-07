# # -- import packages: ----------------------------------------------------------
# import anndata
# import pandas as pd
# # <<<<<<< backend
# # import ABCParse
# # import tqdm.notebook

# # =======
# # import tqdm
# # >>>>>>> pl-smooth-expr

# # -- import local dependencies: ------------------------------------------------
# from ..core import utils
# from ._grouped_expression import GroupedExpression


# # -- set typing: ---------------------------------------------------------------
# from typing import Union, List, Dict, Optional

# from typing import List


# class SmoothedExpressionSummary(ABCParse.ABCParse):
#     def __init__(
#         self,
#         keys: List[str] = ["mean", "std"],
#         storage_key: str = "t_smoothed_gex",
#         *args,
#         **kwargs,
#     ):

#         self.__parse__(locals(), public=[None])
#         self._INFO = utils.InfoMessage()

#     def _isolate_per_gene(self, key: str):
#         """
#         key:
#             "mean" or "std"
#         """
#         return pd.DataFrame(
#             {
#                 gene: values[key].tolist()
#                 for gene, values in self._adata_sim.uns[self._storage_key].items()
#             }
#         )

#     def forward(self):

#         return {
#             f"X_smoothed_gex_{key}": self._isolate_per_gene(key) for key in self._keys
#         }

#     def _add_to_adata(self, outputs):

#         for key, val in outputs.items():
#             description = key.split("_")[-1]
#             self._adata_sim.uns[key] = val
#             self._INFO(
#                 f"Smoothed {description} expression added to: `adata_sim.uns['{key}']`"
#             )

#     def __call__(self, adata_sim, *args, **kwargs):

#         self.__update__(locals(), public=[None])

#         self._add_to_adata(self.forward())


# def summarize_smoothed_expression(
#     adata_sim: anndata.AnnData,
#     keys: List[str] = ["mean", "std"],
#     storage_key: str = "t_smoothed_gex",
#     *args,
#     **kwargs,
# ) -> None:

#     """
#     Parameters
#     ----------
#     adata_sim: anndata.AnnData

#     Returns
#     -------
#     None
#     """

#     smoothed_expression_summary = SmoothedExpressionSummary(
#         keys=keys, storage_key=storage_key
#     )
#     smoothed_expression_summary(adata_sim)
    
# # -- controller class: ---------------------------------------------------------
# class SmoothedExpression(ABCParse.ABCParse):
#     def __init__(
#         self,
#         time_key: str = "t",
#         gene_id_key: str = "gene_ids",
#         use_key: str = "X_gene",
#         disable_tqdm: bool = False,
#         *args,
#         **kwargs,
#     ):

#         self.__parse__(locals(), public=[None])

#     @property
#     def _GROUPED_EXPRESSION(self):
#         if not hasattr(self, "_grouped_expr"):
#             self._grouped_expr = GroupedExpression(
#                 adata=self._adata_sim,
#                 gene_id_key=self._gene_id_key,
#                 use_key=self._use_key,
#             )
#         return self._grouped_expr

#     def _to_frame(self):
#         ...

#     def forward(self, gene_id: str):
#         grouped_vals = self._GROUPED_EXPRESSION(gene_id, groupby=self._time_key)[gene_id]
#         grouped_vals = pd.DataFrame({k: v.flatten() for k, v in grouped_vals.items()})
#         mean, std = grouped_vals.mean(0), grouped_vals.std(0)
#         return {gene_id: pd.DataFrame({"mean": mean, "std": std})}

#     @property
#     def _GENE_IDS(self):
        
#         if self._gene_id is None:
#             return self._adata_sim.uns[self._gene_id_key].tolist()
        
#         if isinstance(self._gene_id, str):
#             return [self._gene_id]
        
#         return self._gene_id

#     def _add_to_anndata(self):

        
#         uns_key = f"{self._time_key}_smoothed_gex"
        
#         if hasattr(self, "_suffix"):
#             uns_key = f"{uns_key}_{self._suffix}"

#         if not uns_key in self._adata_sim.uns:
#             self._adata_sim.uns[uns_key] = {}

#         self._adata_sim.uns[uns_key].update(self._Results)
        
#     @property
#     def _GENE_ID_PROGRESS_BAR(self):
#         if self._disable_tqdm:
#             return self._GENE_IDS
#         return tqdm.notebook.tqdm(self._GENE_IDS)

#     def __call__(
#         self,
#         adata_sim: anndata.AnnData,
#         gene_id: Optional[Union[List[str], str]] = None,
#         suffix: Optional[str] = None,
#         return_dict: bool = False,
#         *args,
#         **kwargs,
#     ):

#         self.__update__(locals(), public = [None], ignore = ['gene_id'])
#         self._gene_id = gene_id
        
#         self._Results = {}
# <<<<<<< backend
#         for gene in tqdm.notebook.tqdm(self._GENE_IDS):
# =======
#         for gene in self._GENE_ID_PROGRESS_BAR:
# >>>>>>> pl-smooth-expr
#             self._Results.update(self.forward(gene))

#         self._add_to_anndata()

#         if self._return_dict:
#             return self._Results


# class SmoothedFrameVarmHandler:
#     """
#     Operating class to mediate adding smoothed expression matrices to adata.varm
#     in the case that all genes are passed.
#     """
#     def __init__(self):
#         ...

#     def _to_frame(self, adata, key):
#         return pd.DataFrame(
#             {gene: expr[key] for gene, expr in adata.uns["t_smoothed_gex"].items()}
#         )

#     @property
#     def mean(self):
#         if not hasattr(self, "_mean"):
#             self._mean = self._to_frame(self._adata, key="mean")
#         return self._mean

#     @property
#     def std(self):
#         if not hasattr(self, "_std"):
#             self._std = self._to_frame(self._adata, key="std")
#         return self._std

#     def __call__(self, adata: anndata.AnnData, return_dfs: bool = False):

#         self._adata = adata

#         self._adata.varm["smoothed_mean"] = self.mean.T
#         self._adata.varm["smoothed_std"] = self.std.T

#         if return_dfs:
#             return self.mean, self.std

# # -- API-facing function: ------------------------------------------------------
# def smoothed_expression(
#     adata_sim: anndata.AnnData,
#     gene_id: Optional[Union[List[str], str]] = None,
#     time_key: str = "t",
#     gene_id_key: str = "gene_ids",
#     use_key: str = "X_gene",
#     return_dict: bool = False,
#     key_added_suffix: str = None,
#     *args,
#     **kwargs,
# ):
    
#     """
#     Parameters
#     ----------
#     adata_sim: anndata.AnnData
#         Simulated AnnData object.
        
#     gene_id: Optional[Union[List[str], str]], default = None
#         Gene name. If None, all genes are used. Called from adata_sim.var_names
        
#         ...
#     """
    
#     if gene_id is None:
#         gene_id = adata_sim.var_names.tolist()

#     smoothed_expression = SmoothedExpression(
#         time_key=time_key, gene_id_key=gene_id_key, use_key=use_key
#     )
# <<<<<<< backend
#     result = smoothed_expression(
#         adata_sim=adata_sim,
#         gene_id=gene_id,
#         suffix = key_added_suffix,
#         return_dict=return_dict,
#     )
    
#     storage_key = f"{time_key}_smoothed_gex"
#     if not key_added_suffix is None:
#         storage_key = "_".join([storage_key, key_added_suffix])
    
#     summarize_smoothed_expression(
#         adata_sim = adata_sim,
#         storage_key = storage_key,
#         keys = ["mean", "std"],
#     )

# =======
#     result = smoothed_expression(adata_sim = adata_sim, gene_id=gene_id, return_dict=return_dict)
    
#     if len(gene_id) == adata_sim.shape[1]:
#         varm_handler = SmoothedFrameVarmHandler()
#         varm_handler(adata_sim, return_dfs = False)
    
# >>>>>>> pl-smooth-expr
#     if return_dict:
#         return result
