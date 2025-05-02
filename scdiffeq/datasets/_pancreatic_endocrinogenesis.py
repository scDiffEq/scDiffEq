# -- import packages: ---------------------------------------------------------
import ABCParse
import anndata
import logging
import os
import pathlib

# -- import local dependencies: -----------------------------------------------
from ._figshare_downloader import figshare_downloader

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)


# -- preprocessing dataset cls: -----------------------------------------------
class UnProcessedPancreaticEndocrinogenesisDataset(ABCParse.ABCParse):
    FIGSHARE_ID = "54151331"
    RAW_FNAME = "_downloaded.pancreas.h5ad"
    PROCESSED_FNAME = "pancreas.pp.h5ad"

    def __init__(self, data_dir=os.getcwd(), *args, **kwargs):
        self.__parse__(locals())

        if not self.data_dir.exists():
            self.data_dir.mkdir()

    @property
    def _scdiffeq_parent_data_dir(self):
        path = pathlib.Path(self._data_dir).joinpath("scdiffeq_data")
        if not path.exists():
            path.mkdir()
        return path

    @property
    def data_dir(self):
        path = self._scdiffeq_parent_data_dir.joinpath("pancreatic_endocrinogenesis")
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
        figshare_downloader(
            figshare_id=self.FIGSHARE_ID,
            write_path=str(self.raw_h5ad_path),
            url_prefix="files",
        )

    @property
    def raw_adata(self):
        if not hasattr(self, "_raw_adata"):
            if not self.raw_h5ad_path.exists():
                self.download_raw()

            raw_adata = anndata.read_h5ad(self.raw_h5ad_path)
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

    @property
    def pp_adata(self):
        if not hasattr(self, "_pp_adata"):
            if not self.processed_h5ad_path.exists():
                self._pp_adata = self.get()
            else:
                self._pp_adata = anndata.read_h5ad(self.processed_h5ad_path)
        return self._pp_adata

    def pp(self, adata, min_genes=200, min_cells=3):

        from .. import scanpy as sc

        adata = adata.copy()

        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata)
        adata = adata[:, adata.var["highly_variable"]].copy()

        return adata

    def dimension_reduction(self, adata: anndata.AnnData):

        from .. import io
        import sklearn
        import umap

        adata = adata.copy()

        self.SCALER_MODEL = sklearn.preprocessing.StandardScaler()
        self.PCA_MODEL = sklearn.decomposition.PCA(n_components=50)
        self.UMAP_MODEL = umap.UMAP(
            n_components=2, n_neighbors=30, random_state=0, min_dist=0.5
        )

        adata.obsm["X_scaled"] = self.SCALER_MODEL.fit_transform(adata.X.toarray())
        adata.obsm["X_pca"] = self.PCA_MODEL.fit_transform(adata.obsm["X_scaled"])
        adata.obsm["X_umap"] = self.UMAP_MODEL.fit_transform(adata.obsm["X_pca"])

        io.write_pickle(
            self.SCALER_MODEL,
            self.data_dir.joinpath("pancreatic_endocrinogenesis.pp.scaler_model.pkl"),
        )
        io.write_pickle(
            self.PCA_MODEL,
            self.data_dir.joinpath("pancreatic_endocrinogenesis.pp.pca_model.pkl"),
        )
        io.write_pickle(
            self.UMAP_MODEL,
            self.data_dir.joinpath("pancreatic_endocrinogenesis.pp.umap_model.pkl"),
        )

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
        adata_pp = self.dimension_reduction(adata_pp)
        adata = self.annotate_time(adata_pp)
        adata.write_h5ad(self.processed_h5ad_path)
        return adata


def _unprocessed_pancreas(data_dir: str = os.getcwd()) -> anndata.AnnData:
    """Pancreas dataset

    Parameters
    ----------
    data_dir : str, default=os.getcwd()
        Path to the directory where the data will be saved.

    Returns
    -------
    anndata.AnnData
        Processed AnnData object.
    """
    data_handler = UnProcessedPancreaticEndocrinogenesisDataset(data_dir=data_dir)
    return data_handler.pp_adata


# -- dataset cls: -------------------------------------------------------------
class PancreaticEndocrinogenesisDataset(ABCParse.ABCParse):
    _FNAME = "pancreatic_endocrinogenesis"
    _ADATA_FNAME = f"adata.{_FNAME}.cytotrace.h5ad"
    figshare_ids = {
        "adata": {"file_id": 54151208, "fname": _ADATA_FNAME},
        "scaler": {"file_id": 54151202, "fname": f"{_FNAME}.scaler.pkl"},
        "pca": {"file_id": 54151205, "fname": f"{_FNAME}.pca.pkl"},
        "umap": {"file_id": 54151199, "fname": f"{_FNAME}.umap.pkl"},
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
        path = self._scdiffeq_parent_data_dir.joinpath("pancreatic_endocrinogenesis")
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
            if not self.h5ad_path.exists() or self._force_download:
                self.download()
            self._adata = anndata.read_h5ad(self.h5ad_path)
        return self._adata


def pancreatic_endocrinogenesis(
    data_dir: str = os.getcwd(),
    filter_genes: bool = True,
    reduce_dimensions: bool = True,
    force_download: bool = False,
    download_unprocessed: bool = False,
) -> anndata.AnnData:
    """Pancreas dataset

    Args:
        data_dir: str, default=os.getcwd()
            Path to the directory where the data will be saved.
        filter_genes: bool, default=True
            Whether to filter genes.
        reduce_dimensions: bool, default=True
            Whether to reduce dimensions.
        force_download: bool, default=False
            Whether to force download the data.
        download_unprocessed: bool, default=False
            Whether to download the unprocessed data.

    Returns:
        anndata.AnnData: Preprocessed AnnData object.
    """
    if download_unprocessed:
        return _unprocessed_pancreas(
            data_dir=data_dir,
            filter_genes=filter_genes,
            reduce_dimensions=reduce_dimensions,
            force_download=force_download,
        )
    else:
        data_handler = PancreaticEndocrinogenesisDataset(
            data_dir=data_dir,
            force_download=force_download,
        )
        return data_handler.adata
