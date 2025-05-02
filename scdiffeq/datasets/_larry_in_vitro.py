# -- import packages: ----------------------------------------------------------
import ABCParse
import anndata
import logging
import os
import pathlib
import sklearn

# -- import local dependencies: ------------------------------------------------
from .. import io
from ._figshare_downloader import figshare_downloader

# -- configure logger: ----------------------------------------------------------
logger = logging.getLogger(__name__)


# -- Controller class: ---------------------------------------------------------
class LARRYInVitroDataset(ABCParse.ABCParse):
    FNAME = "larry.h5ad"
    figshare_id = 52612805

    def __init__(
        self,
        data_dir=os.getcwd(),
        filter_genes: bool = True,
        reduce_dimensions: bool = True,
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
            if self._filter_genes:
                adata = self._gene_filtering(adata)
            if self._reduce_dimensions:
                self._dimension_reduction(adata)
            adata.write_h5ad(self.h5ad_path)
        return adata

    @property
    def adata(self) -> anndata.AnnData:
        if not hasattr(self, "_adata"):
            if not self.h5ad_path.exists() or self._force_download:
                self.download()
                adata = anndata.read_h5ad(self.h5ad_path)
                self._adata = self._preprocess(adata=adata)
                return self._adata
            else:
                logger.info(f"Loading data from {self.h5ad_path}")
                return anndata.read_h5ad(self.h5ad_path)


def larry(
    data_dir: str = os.getcwd(),
    filter_genes: bool = True,
    reduce_dimensions: bool = True,
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
        force_download: bool, default=False
            Whether to force download the data.

    Returns:
        anndata.AnnData: Preprocessed AnnData object.
    """
    data_handler = LARRYInVitroDataset(
        data_dir=data_dir,
        filter_genes=filter_genes,
        reduce_dimensions=reduce_dimensions,
        force_download=force_download,
    )
    return data_handler.adata
