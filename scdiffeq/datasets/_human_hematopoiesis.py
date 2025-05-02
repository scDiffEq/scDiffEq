# -- import packages: ---------------------------------------------------------
import ABCParse
import anndata
import logging
import os
import numpy as np
import pathlib


# -- import local dependencies: -----------------------------------------------
from ._figshare_downloader import figshare_downloader

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)


# -- preprocessing dataset cls: -----------------------------------------------
class UnProcessedHumanHematpoiesisDataHandler(ABCParse.ABCParse):
    # HTTPS = "https://www.dropbox.com/s/8m8n6fj8yn1ryjd/hsc_all_combined_all_layers.h5ad?dl=1"
    RAW_FNAME = "_hsc_all_combined_all_layers.h5ad"
    PROCESSED_FNAME = "human_hematopoiesis.processed.h5ad"

    def __init__(
        self,
        data_dir=os.getcwd(),
        skip_scaling: bool = False,
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
        path = self._scdiffeq_parent_data_dir.joinpath("human_hematopoiesis")
        if not path.exists():
            path.mkdir()
        return path

    @property
    def raw_h5ad_path(self) -> pathlib.Path:
        return self.data_dir.joinpath(self.RAW_FNAME)

    @property
    def processed_h5ad_path(self) -> pathlib.Path:
        return self.data_dir.joinpath(self.PROCESSED_FNAME)

    def download_raw(self):
        if not self.raw_h5ad_path.exists():
            figshare_downloader(
                figshare_id=54154235,
                write_path=self.raw_h5ad_path,
            )

    @property
    def raw_adata(self):
        if not hasattr(self, "_raw_adata"):
            if not self.raw_h5ad_path.exists():
                self.download_raw()
            self._raw_adata = anndata.read_h5ad(self.raw_h5ad_path)
            self._raw_idx = self._raw_adata.obs.index
        return self._raw_adata

    @property
    def pp_adata(self):
        if not hasattr(self, "_pp_adata"):
            if not self.processed_h5ad_path.exists():
                self._pp_adata = self.get(skip_scaling=self._skip_scaling)
            else:
                self._pp_adata = anndata.read_h5ad(self.processed_h5ad_path)
        return self._pp_adata

    def seurat_preprocessing(self, adata: anndata.AnnData):

        import scanpy as sc

        adata = adata.copy()

        adata.obs["time"] = 3
        adata.obs.loc[adata.obs.batch == "JL_10", "time"] = 5
        adata.obs.time.value_counts()
        adata.obs_names_make_unique()
        adata.obs["nGenes"] = (adata.X > 0).sum(1)
        adata.obs["UMI"] = (adata.X).sum(1)
        adata.layers["X_orig"] = adata.X
        adata.X = adata.layers["labeled_TC"].copy()
        # Convert pandas Series to numpy array for proper boolean indexing
        jl_10_mask = (adata.obs.batch == "JL_10").values
        adata.X[jl_10_mask, :] *= 3 / 5
        adata.obs.rename({"time": "_time"}, axis=1, inplace=True)
        t_map = {"JL_10": 4, "JL12_0": 7, "JL12_1": 7}
        adata.obs["t"] = adata.obs["batch"].map(t_map)
        sc.pp.recipe_seurat(adata)

        return adata

    def _match_and_filter_on_idx(self, adata: anndata.AnnData):
        """Match and filter on idx."""

        # HTTPS = "https://www.dropbox.com/s/n9mx9trv1h78q0r/hematopoiesis_v1.h5ad?dl=1"
        _secondary_h5ad = self.data_dir.joinpath("_dynamo_hematopoiesis_v1.h5ad")
        if not _secondary_h5ad.exists():
            figshare_downloader(
                figshare_id=54154238,
                write_path=_secondary_h5ad,
            )

        adata_dyn = anndata.read_h5ad(_secondary_h5ad)

        match_idx = np.where(adata.obs.index.isin(adata_dyn.obs.index))
        adata = adata[match_idx].copy()
        adata.obs["cell_type"] = adata_dyn[adata.obs.index].obs["cell_type"].values

        return adata

    def dimension_reduction(self, adata: anndata.AnnData, skip_scaling: bool = True):

        import sklearn
        import umap

        from .. import io

        adata = adata.copy()

        self.SCALER_MODEL = sklearn.preprocessing.StandardScaler()
        self.PCA_MODEL = sklearn.decomposition.PCA(n_components=50)
        self.UMAP_MODEL = umap.UMAP(
            n_components=2,
            n_neighbors=25,
            random_state=1,
            min_dist=0.5,
        )

        if skip_scaling:
            adata.obsm["X_pca"] = self.PCA_MODEL.fit_transform(adata.X)
        else:
            adata.layers["X_scaled"] = self.SCALER_MODEL.fit_transform(adata.X)
            adata.obsm["X_pca"] = self.PCA_MODEL.fit_transform(adata.layers["X_scaled"])
            io.write_pickle(
                self.SCALER_MODEL,
                self.data_dir.joinpath("human_hematopoiesis.scaler.pkl"),
            )

        adata.obsm["X_umap"] = self.UMAP_MODEL.fit_transform(
            adata.obsm["X_pca"][:, :10]
        )

        io.write_pickle(
            self.PCA_MODEL, self.data_dir.joinpath("human_hematopoiesis.pca.pkl")
        )
        io.write_pickle(
            self.UMAP_MODEL,
            self.data_dir.joinpath("human_hematopoiesis.umap.pkl"),
        )

        return self._match_and_filter_on_idx(adata)

    def get(self, skip_scaling: bool = False):

        adata_pp = self.seurat_preprocessing(self.raw_adata)
        adata_pp = self.dimension_reduction(adata_pp, skip_scaling=skip_scaling)
        adata_pp.write_h5ad(self.processed_h5ad_path)
        return adata_pp


# -- dataset cls: -------------------------------------------------------------
class HumanHematopoiesisDataset(ABCParse.ABCParse):
    _FNAME = "human_hematopoiesis"
    _ADATA_FNAME = f"{_FNAME}.processed.h5ad"
    figshare_ids = {
        "adata": {"file_id": 54154232, "fname": _ADATA_FNAME},
        "scaler": {"file_id": 54154226, "fname": f"{_FNAME}.scaler.pkl"},
        "pca": {"file_id": 54154223, "fname": f"{_FNAME}.pca.pkl"},
        "umap": {"file_id": 54154229, "fname": f"{_FNAME}.umap.pkl"},
    }

    def __init__(
        self,
        data_dir=os.getcwd(),
        force_download: bool = False,
        *args,
        **kwargs,
    ):

        self.__parse__(locals())

    @property
    def _scdiffeq_parent_data_dir(self):
        path = pathlib.Path(self._data_dir).joinpath("scdiffeq_data")
        path.mkdir(exist_ok=True, parents=True)
        return path

    @property
    def data_dir(self):
        path = self._scdiffeq_parent_data_dir.joinpath("human_hematopoiesis")
        path.mkdir(exist_ok=True, parents=True)
        return path

    @property
    def h5ad_path(self) -> pathlib.Path:
        return self.data_dir.joinpath(self._ADATA_FNAME)

    def download(self):
        for _, val in self.figshare_ids.items():
            figshare_downloader(
                figshare_id=val["file_id"],
                write_path=self.data_dir.joinpath(val["fname"]),
            )

    @property
    def adata(self) -> anndata.AnnData:
        if not hasattr(self, "_adata"):
            if (not self.h5ad_path.exists()) or self._force_download:
                self.download()
            self._adata = anndata.read_h5ad(self.h5ad_path)
        return self._adata


# -- download function for unprocessed data: -----------------------------------
def _download_unprocessed_human_hematopoiesis(
    data_dir=os.getcwd(),
    skip_scaling: bool = False,
):
    _handler = UnProcessedHumanHematpoiesisDataHandler(
        data_dir=data_dir,
        skip_scaling=skip_scaling,
    )
    return _handler.pp_adata


# -- main api-facing function: ------------------------------------------------
def human_hematopoiesis(
    data_dir: str = os.getcwd(),
    skip_scaling: bool = False,
    force_download: bool = False,
    download_unprocessed: bool = False,
) -> anndata.AnnData:
    """Human hematopoiesis dataset

    Args:
        data_dir: str, default=os.getcwd()
            Path to the directory where the data will be saved.
        skip_scaling: bool, default=False
            Whether to skip scaling.
        force_download: bool, default=False
            Whether to force download the data.
        download_unprocessed: bool, default=False
            Whether to download the unprocessed data.

    Returns:
        anndata.AnnData: Preprocessed AnnData object.
    """
    if download_unprocessed:
        return _download_unprocessed_human_hematopoiesis(
            data_dir=data_dir,
            skip_scaling=skip_scaling,
        )
    else:
        data_handler = HumanHematopoiesisDataset(
            data_dir=data_dir,
            force_download=force_download,
        )
        return data_handler.adata
