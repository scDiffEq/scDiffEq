
# -- import packages: ----------------------------------------------------------
import pandas as pd
import numpy as np
import sklearn
import anndata
import umap
import ABCParse

# -- import local dependencies: ------------------------------------------------
from ..core import utils
from .. import _backend_utilities as backend

# -- set typing: ---------------------------------------------------------------
from typing import Optional, Union, List

NoneType = type(None)


# -- Operator class: -----------------------------------------------------------
class Perturbation(ABCParse.ABCParse):
    """
    Alter the relative expression of one or more genes in one
    or more cells.
    """
    def __init__(
        self,
        adata: anndata.AnnData,
        gene_id_key: str = "gene_ids",
        use_key: str = "X_scaled",
        PCA: Optional[sklearn.decomposition.PCA] = None,
        UMAP: Optional[umap.UMAP] = None,
        copy: bool = False,
    ) -> NoneType:
        
        """ """
        
        self.__parse__(locals(), ignore = ['adata'], public = [None])
        
        if self._copy:
            self.adata = adata.copy()
        else:
            self.adata = adata
        
        self.obs_lookup = backend.ObsIndexLookUp()
        self.var_lookup = backend.VarIndexLookUp(var_name_key = gene_id_key)

    @property
    def X(self) -> np.ndarray:
        if not hasattr(self, "_X"):
            self._X = self.adata.layers[self._use_key].copy()
        return self._X

    @property
    def X_pca_perturbed(self):
        if not hasattr(self, "_X_pca_perturbed"):
            self._X_pca_perturbed = self._PCA.transform(self.X_perturb[self.cell_idx])
        return self._X_pca_perturbed

    @property
    def X_umap_perturbed(self):
        if not hasattr(self, "_X_umap_perturbed"):
            self._X_umap_perturbed = self._UMAP.transform(self.X_pca_perturbed)
        return self._X_umap_perturbed

    @property
    def obs_perturbed(self):
        perturb_indicator = np.zeros(self.adata.shape[0], dtype=bool)
        perturb_indicator[self.cell_idx] = True
        return perturb_indicator

    @property
    def var_perturbed(self):
        perturb_indicator = np.zeros(self.adata.shape[1], dtype=bool)
        perturb_indicator[self.gene_idx] = True
        return perturb_indicator
        
    def _update_adata(self):
        
        self.adata.uns["X_perturb"] = self.X_perturb[self.cell_idx]
        
        if not isinstance(self._PCA, NoneType):
            self.adata.uns["X_pca_perturb"] = self.X_pca_perturbed
        if not isinstance(self._UMAP, NoneType):
            self.adata.uns["X_umap_perturb"] = self.X_umap_perturbed
        
        self.adata.obs["perturbed"] = self.obs_perturbed
        self.adata.var["perturbed"] = self.var_perturbed
        
        self.adata.uns['perturbed'] = {
            "genes": self._gene_idx,
            "cells": self._cell_idx,
        }
        
        
    def forward(self, cell_idx: Union[int, List], gene_idx: int, val: float):
        """ """
        self.X_perturb[cell_idx, gene_idx] = val
        
    @property
    def cell_idx(self):
        return self.obs_lookup(self.adata, self._cell_idx)
    
    @property
    def gene_idx(self):
        return self.var_lookup(self.adata, self._gene_idx)
    
    def _prepare_return(self):
        
        return_ = []
        if self.return_array:
            return_.append(self.X_perturb)
        if self._copy:
            return_.append(self.adata)
        
        if len(return_) > 0:
            return return_
        
        
    def __call__(
        self,
        cell_idx: Union[int, List],
        gene_idx: List,
        val: float,
        return_array: bool = False,
    ):
        """ """
        
        self.__update__(locals(), private = ['cell_idx', 'gene_idx'])
        
        self.X_perturb = self.X
        
        for gene_ix in self.gene_idx:
            self.forward(self.cell_idx, gene_ix, val)
        
        self._update_adata()
            
        return self._prepare_return()


# -- API-facing function: ------------------------------------------------------
def perturb(
    adata: anndata.AnnData,
    cell_idx: Union[pd.Index, np.array, list],
    gene_idx: Union[pd.Index, np.array, list],
    val: float,
    use_key: str = "X_scaled",
    PCA: Optional[sklearn.decomposition.PCA] = None,
    UMAP: Optional[umap.UMAP] = None,
    return_array: bool = False,
    copy: bool = False,
) -> anndata.AnnData:
    
    """
    Impart a perturbation.
    
    Parameters:
    -----------
    adata
        Annotated Data object.
        type: anndata.AnnData
        
    cell_idx
        Cells to be perturbed. Should correspond to the integer indices of the layer
        chosen by adata.layers[use_key].
        type: Union[pd.Index, np.array, list]
        
    gene_idx
        Genes to be perturbed. Should correspond to the integer indices of the layer
        chosen by adata.layers[use_key].
        type: Union[pd.Index, np.array, list]
    
    val
        Modified expression value
        type: float

    use_key
        Layer from which reference gene expression should be taken.
        type: str
        default: "X_scaled"
        
    PCA
        PCA model
        type: Optional[sklearn.decomposition.PCA]
        default: None
        
    UMAP
        UMAP model
        type: Optional[umap.UMAP]
        default: None
        
    copy
        If `True`, returns a copy of the input `adata`.
        type: bool
        default: False
        
    return_array
        type: bool
        default: False
        

    Returns:
    --------
    adata_perturbed
    """
    
    perturbation = Perturbation(
        adata = adata,
        PCA = PCA,
        UMAP = UMAP,
        use_key = use_key,
        copy = copy,
    )
    return perturbation(
        cell_idx = cell_idx,
        gene_idx = gene_idx,
        val = val,
        return_array = return_array,
    )
    