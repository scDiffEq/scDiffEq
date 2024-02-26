import ABCParse
import pathlib
import anndata
import scanpy as sc
import os
import sklearn
import umap
import numpy as np
import web_files


from .. import io

class HumanHematpoiesisDataHandler(ABCParse.ABCParse):
    HTTPS = "https://www.dropbox.com/s/8m8n6fj8yn1ryjd/hsc_all_combined_all_layers.h5ad?dl=1"
    FNAME = "hsc_all_combined_all_layers.h5ad"
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
            self._raw_adata = anndata.read_h5ad(self.h5ad_path)
            self._raw_idx = self._raw_adata.obs.index
        return self._raw_adata    

    def seurat_preprocessing(self, adata):

        adata = adata.copy()

        adata.obs['time'] = 3
        adata.obs.loc[adata.obs.batch == 'JL_10', 'time'] = 5
        adata.obs.time.value_counts()
        adata.obs_names_make_unique()
        adata.obs["nGenes"] = (adata.X > 0).sum(1)
        adata.obs["UMI"] = (adata.X).sum(1)
        adata.layers['X_orig'] = adata.X
        adata.X = adata.layers['labeled_TC'].copy()
        adata.X[adata.obs.batch == 'JL_10', :] *= 3/5
        adata.obs.rename({"time": "_time"}, axis = 1, inplace=True)
        t_map = {'JL_10': 4, 'JL12_0': 7, 'JL12_1': 7}
        adata.obs['t'] = adata.obs['batch'].map(t_map)
        sc.pp.recipe_seurat(adata)

        return adata
    
    def _match_and_filter_on_idx(self, adata):

        """"""

        HTTPS = "https://www.dropbox.com/s/n9mx9trv1h78q0r/hematopoiesis_v1.h5ad?dl=1"
        _secondary_h5ad = self.data_dir.joinpath("hematopoiesis_v1.h5ad")
        if not _secondary_h5ad.exists():
            web_file = web_files.WebFile(http_address=HTTPS, local_path=str(_secondary_h5ad))
            web_file.download()  
        
        adata_dyn = anndata.read_h5ad(_secondary_h5ad)
        
        match_idx = np.where(adata.obs.index.isin(adata_dyn.obs.index))
        adata = adata[match_idx].copy()
        adata.obs['cell_type'] = adata_dyn[adata.obs.index].obs['cell_type'].values

        return adata

    
    def dimension_reduction(self, adata: anndata.AnnData, skip_scaling: bool = True):
    
        adata = adata.copy()

        self.SCALER_MODEL = sklearn.preprocessing.StandardScaler()
        self.PCA_MODEL = sklearn.decomposition.PCA(n_components=50)
        self.UMAP_MODEL = umap.UMAP(n_components=2, n_neighbors=25, random_state=1, min_dist= 0.5,)

        if skip_scaling:
            adata.obsm['X_pca'] = self.PCA_MODEL.fit_transform(adata.X)
        else:
            adata.layers['X_scaled'] = self.SCALER_MODEL.fit_transform(adata.X)
            adata.obsm['X_pca'] = self.PCA_MODEL.fit_transform(adata.layers['X_scaled'])
            io.write_pickle(self.SCALER_MODEL, self.data_dir.joinpath("human_hematopoiesis.scaler_model.pkl"))

        adata.obsm['X_umap'] = self.UMAP_MODEL.fit_transform(adata.obsm['X_pca'][:, :10])        
        
        io.write_pickle(self.PCA_MODEL, self.data_dir.joinpath("human_hematopoiesis.pca_model.pkl"))
        io.write_pickle(self.UMAP_MODEL, self.data_dir.joinpath("human_hematopoiesis.umap_model.pkl"))
        
        return self._match_and_filter_on_idx(adata)
    
    def get(self):
        
        adata_pp = self.seurat_preprocessing(self.raw_adata)
        return self.dimension_reduction(adata_pp)
    
def human_hematopoiesis():
    """
    Human hematopoiesis dataset from Dynamo.
    
    For more, see:
    https://dynamo-release.readthedocs.io/en/latest/notebooks/tutorial_hsc_velocity.html"""
    data_handler = HumanHematpoiesisDataHandler()
    return data_handler.get()
