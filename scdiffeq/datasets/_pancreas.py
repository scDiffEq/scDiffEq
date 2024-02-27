

# -- import packages: ----------------------------------------------------------
import ABCParse
import pathlib
import anndata
import scanpy as sc
import os
import sklearn
import umap
import numpy as np
import web_files


# -- import local dependencies: ------------------------------------------------
from .. import io


# -- set typing: ---------------------------------------------------------------
from typing import Union


# -- Controller class: ---------------------------------------------------------
class PancreaticEndocrinogenesisDataset(ABCParse.ABCParse):
    HTTPS = "https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad"
    FNAME = "pancreas.h5ad"
    def __init__(self, data_dir = os.getcwd(), *args, **kwargs):
        self.__parse__(locals())
        
        if not self.data_dir.exists():
            self.data_dir.mkdir()
        
    @property
    def data_dir(self):
        return pathlib.Path(self._data_dir).joinpath("scdiffeq_data")
        
    @property
    def h5ad_path(self) -> pathlib.Path:
        return self.data_dir.joinpath(self.FNAME)
    
    def download(self):
        if not self.h5ad_path.exists():
            web_file = web_files.WebFile(http_address=self.HTTPS, local_path=str(self.h5ad_path))
            web_file.download()            
            
    @property
    def raw_adata(self):
        if not hasattr(self, "_adata"):
            self.download()
            raw_adata = anndata.read_h5ad(self.h5ad_path)
            self._raw_idx = raw_adata.obs.index
            
            del raw_adata.uns["pca"]
            del raw_adata.uns["neighbors"]
            del raw_adata.obsm["X_pca"]
            del raw_adata.obsm["X_umap"]
            del raw_adata.obsp["distances"]
            del raw_adata.obsp["connectivities"]
            
            raw_adata.layers["X_counts"] = raw_adata.X
            
            raw_adata.obs = raw_adata.obs.reset_index()
            raw_adata.obs.index = raw_adata.obs.index.astype(str)
            
            self._raw_adata = raw_adata
            
        return self._raw_adata    

    def pp(self, adata, min_genes=200, min_cells = 3):
        
        adata = adata.copy()
        
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata)
        adata = adata[:, adata.var["highly_variable"]].copy()
        
        return adata
    
    def dimension_reduction(self, adata: anndata.AnnData):
    
        adata = adata.copy()
        
        self.SCALER_MODEL = sklearn.preprocessing.StandardScaler()
        self.PCA_MODEL = sklearn.decomposition.PCA(n_components=50)
        self.UMAP_MODEL = umap.UMAP(n_components=2, n_neighbors=30, random_state=0, min_dist=0.5)
        
        adata.obsm["X_scaled"] = self.SCALER_MODEL.fit_transform(adata.X.A)
        adata.obsm["X_pca"] = self.PCA_MODEL.fit_transform(adata.obsm["X_scaled"])
        adata.obsm["X_umap"] = self.UMAP_MODEL.fit_transform(adata.obsm["X_pca"])

        io.write_pickle(self.SCALER_MODEL, self.data_dir.joinpath("pancreas.scaler_model.pkl"))        
        io.write_pickle(self.PCA_MODEL, self.data_dir.joinpath("pancreas.pca_model.pkl"))
        io.write_pickle(self.UMAP_MODEL, self.data_dir.joinpath("pancreas.umap_model.pkl"))
        
        return adata
    
    
    def annotate_time(self, adata):

        idx = adata.obs.index
        t0_str_idx = adata.obs[adata.obs["clusters"] == "Ductal"].index
        t0_idx = idx.isin(t0_str_idx)
        t1_idx = ~idx.isin(t0_str_idx)
        t = np.zeros(len(idx))
        t[t1_idx] = 1
        adata.obs["t"] = t
        
        return adata
    
    def get(self):
        
        adata_pp = self.pp(self.raw_adata)
        adata = self.dimension_reduction(adata_pp)
        return self.annotate_time(adata)
    
def pancreas():
    """
    Pancreas dataset
    
    For more, see:
    """
    data_handler = PancreaticEndocrinogenesisDataset()
    return data_handler.get()
