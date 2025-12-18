# -- import packages: ----------------------------------------------------------
import ABCParse
import anndata
import logging
import os
import pandas as pd
import pathlib
import sklearn

# -- import local dependencies: ------------------------------------------------
from .. import io
from ._figshare_downloader import figshare_downloader

# -- set type hints: -----------------------------------------------------------
from typing import Dict, Union

# -- configure logger: ----------------------------------------------------------
logger = logging.getLogger(__name__)


def _annotate_larry_cytotrace(adata: anndata.AnnData, data_dir: Union[str, pathlib.Path]) -> Dict[str, pd.DataFrame]:
    
    """ """
    
    obs_write_path = str(pathlib.Path(data_dir).joinpath("larry.ct_obs_df.csv"))
    var_write_path = str(pathlib.Path(data_dir).joinpath("larry.ct_var_df.csv"))    
    
    figshare_downloader(
        figshare_id="54312011",
        write_path=obs_write_path,
    )
    figshare_downloader(
        figshare_id="54312008",
        write_path=var_write_path,
    )
    
    obs_df = pd.read_csv(obs_write_path, index_col = 0)
    var_df = pd.read_csv(var_write_path, index_col = 0)

    # Convert indices to strings
    obs_df.index = obs_df.index.astype(str)
    var_df.index = var_df.index.astype(str)
    
    # Convert all columns to strings
    for col in obs_df.columns:
        obs_df[col] = obs_df[col].astype(str)
    for col in var_df.columns:
        var_df[col] = var_df[col].astype(str)

    adata.obs = pd.concat([adata.obs, obs_df], axis = 1)
    adata.var = pd.concat([adata.var, var_df], axis = 1)

    return adata

# -- Controller class: ---------------------------------------------------------
class LARRYInVitroDataset(ABCParse.ABCParse):
    FNAME = "larry.h5ad"
    figshare_id = 52612805

    def __init__(
        self,
        data_dir=os.getcwd(),
        filter_genes: bool = True,
        reduce_dimensions: bool = True,
        cytotrace: bool = True,
        force_download: bool = False,
        *args,
        **kwargs,
    ):

        self.__parse__(locals())

    @property
    def _scdiffeq_parent_data_dir(self):
        path = pathlib.Path(self._data_dir).joinpath("scdiffeq_data")
        if not path.exists():
            path.mkdir()
        return path

    @property
    def data_dir(self):
        path = self._scdiffeq_parent_data_dir.joinpath("larry")
        if not path.exists():
            path.mkdir()
        return path

    @property
    def h5ad_path(self) -> pathlib.Path:
        return self.data_dir.joinpath(self.FNAME)

    @property
    def _DO_PREPROCESSING(self):
        return any([self._filter_genes, self._reduce_dimensions])

    def download(self):
        figshare_downloader(
            figshare_id=self.figshare_id,
            write_path=self.h5ad_path,
        )

    def _gene_filtering(self, adata: anndata.AnnData) -> anndata.AnnData:
        return adata[:, adata.var["use_genes"]].copy()

    def _dimension_reduction(self, adata: anndata.AnnData):
        """Do sample dimension reduction"""
        # -- instantiate models: ----------------------------------------------
        scaler = sklearn.preprocessing.StandardScaler()
        pca = sklearn.decomposition.PCA(n_components=50)

        # -- fit transform data: ----------------------------------------------
        adata.obsm["X_scaled"] = scaler.fit_transform(adata.X.toarray())
        adata.obsm["X_pca"] = pca.fit_transform(adata.obsm["X_scaled"])

        # -- save models: -----------------------------------------------------
        io.write_pickle(
            obj=scaler,
            path=self.data_dir.joinpath("scaler.pkl"),
        )
        io.write_pickle(
            obj=pca,
            path=self.data_dir.joinpath("pca.pkl"),
        )

    def _preprocess(self, adata: anndata.AnnData) -> anndata.AnnData:
        if self._DO_PREPROCESSING:
            logger.info("Preprocessing...")
            if self._cytotrace:
                _annotate_larry_cytotrace(adata=adata, data_dir=self.data_dir)
            if self._filter_genes:
                adata = self._gene_filtering(adata)
            if self._reduce_dimensions:
                self._dimension_reduction(adata)
            adata.write_h5ad(self.h5ad_path)
            self._adata = adata
        
    
    def _safe_read(self):
        logger.info(f"Loading data from {self.h5ad_path}")
        try:
            adata = anndata.read_h5ad(self.h5ad_path)
            if "ct_pseudotime" in adata.obs.columns:
                adata.obs['ct_pseudotime'] = adata.obs['ct_pseudotime'].astype(float)
            adata.obs.index.name = "index"
            self._adata = adata
        except Exception as e:
            logger.error(f"Error loading data from {self.h5ad_path}: {e}")
            raise e
        return adata

    @property
    def adata(self) -> anndata.AnnData:
        if not hasattr(self, "_adata"):
            if not self.h5ad_path.exists() or self._force_download:
                self.download()
                adata = anndata.read_h5ad(self.h5ad_path)
                self._preprocess(adata=adata)
            return self._safe_read()


def larry(
    data_dir: str = os.getcwd(),
    filter_genes: bool = True,
    reduce_dimensions: bool = True,
    cytotrace: bool = True,
    force_download: bool = False,
) -> anndata.AnnData:
    """LARRY in vitro dataset

    Args:
        data_dir: str, default=os.getcwd()
            Path to the directory where the data will be saved.
        filter_genes: bool, default=True
            Whether to filter genes.
        reduce_dimensions: bool, default=True
            Whether to reduce dimensions.
        cytotrace: bool, default=True
            Whether to annotate LARRY with pre-computed CytoTRACE annotations.
        force_download: bool, default=False
            Whether to force download the data.

    Returns:
        anndata.AnnData: Preprocessed AnnData object.
    """
    data_handler = LARRYInVitroDataset(
        data_dir=data_dir,
        filter_genes=filter_genes,
        reduce_dimensions=reduce_dimensions,
        cytotrace=cytotrace,
        force_download=force_download,
    )
    return data_handler.adata
