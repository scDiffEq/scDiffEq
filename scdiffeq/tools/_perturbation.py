
# -- import packages: ----------------------------------------------------------
import pandas as pd
import numpy as np
import sklearn
import anndata
import umap


# -- import local dependencies: ------------------------------------------------
from ..core import utils


# -- set typing: ---------------------------------------------------------------
from typing import Optional, Union, List

NoneType = type(None)


# -- Operator class: -----------------------------------------------------------
class Perturbation(utils.ABCParse):
    """
    Alter the relative expression of one or more genes in one
    or more cells.
    """
    def __init__(
        self,
        adata: anndata.AnnData,
        use_key: str = "X_scaled",
        PCA: Union[Optional, sklearn.decomposition.PCA] = None,
        UMAP: Union[Optional, umap.UMAP] = None,
    ) -> NoneType:
        
        """ """
        
        self.__parse__(locals(), ignore = ['adata'], public = [None])
        self.adata = adata.copy()

    @property
    def X(self) -> np.ndarray:
        if not hasattr(self, "_X"):
            self._X = self._adata.layers[self._use_key].copy()
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
        perturb_indicator = np.zeros(self._adata.shape[0], dtype=bool)
        perturb_indicator[self.cell_idx] = True
        return perturb_indicator

    @property
    def var_perturbed(self):
        perturb_indicator = np.zeros(self._adata.shape[1], dtype=bool)
        perturb_indicator[self.gene_idx] = True
        return perturb_indicator
        
    def _update_adata(self):
        
        self.adata.uns["X_perturb"] = self.X_perturb[self.cell_idx]
        
        if hasattr(self, "_PCA"):
            self.adata.uns["X_pca_perturb"] = self.X_pca_perturbed
        if hasattr(self, "_UMAP"):
            self.adata.uns["X_umap_perturb"] = self.X_umap_perturbed
        
        self._adata.obs["perturbed"] = self.obs_perturbed
        self._adata.var["perturbed"] = self.var_perturbed
        
        
    def forward(self, cell_idx: Union[int, List], gene_idx: int, val: float):
        """ """
        self.X_perturb[cell_idx, gene_idx] = val
        
    def __call__(
        self,
        cell_idx: Union[int, List],
        gene_idx: List,
        val: float,
        return_array: bool = False,
    ):
        """ """
        
        self.__update__(locals())
        
        self.X_perturb = self.X
        
        for gene_ix in gene_idx:
            self.forward(cell_idx, gene_ix, val)
        
        self._update_adata()
        
        if self.return_array:
            return self.X_perturb
        
        return self.adata


# -- API-facing function: ------------------------------------------------------
def perturb(
    adata: anndata.AnnData,
    cell_idx: Union[pd.Index, np.array, list],
    gene_idx: Union[pd.Index, np.array, list],
    val: float,
    use_key: str = "X_scaled",
    PCA: Optional = None,
    UMAP: Optional = None,
    return_array: bool = False,
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
        type: Union[Optional, sklearn.decomposition.PCA]
        default: None
        
    UMAP
        UMAP model
        type: Union[Optional, umap.UMAP]
        default: None
        
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
    )
    return perturbation(
        cell_idx = cell_idx,
        gene_idx = gene_idx,
        val = val,
        return_array = return_array,
    )
    