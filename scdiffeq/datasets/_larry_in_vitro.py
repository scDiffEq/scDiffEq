# -- import packages: ----------------------------------------------------------
import ABCParse
import pathlib
import anndata
import os
import requests


# -- import local dependencies: ------------------------------------------------
from .. import io


# -- Controller class: ---------------------------------------------------------
class LARRYInVitroDataset(ABCParse.ABCParse):
    FNAME = "larry.h5ad"
    figshare_id = 52612805

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
        path = self._scdiffeq_parent_data_dir.joinpath("larry")
        if not path.exists():
            path.mkdir()
        return path

    @property
    def h5ad_path(self) -> pathlib.Path:
        return self.data_dir.joinpath(self.FNAME)

    def download(self):
        url = f"https://figshare.com/ndownloader/files/{self.figshare_id}"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(self.h5ad_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    @property
    def adata(self):
        if not hasattr(self, "_adata"):
            if not self.h5ad_path.exists():
                self.download()

            adata = anndata.read_h5ad(self.h5ad_path)
            adata.obs = adata.obs.reset_index()
            adata.obs.index = adata.obs.index.astype(str)
            self._adata = adata

        return self._adata


def larry(data_dir: str = os.getcwd()) -> anndata.AnnData:
    """LARRY in vitro dataset

    Parameters
    ----------
    data_dir : str, default=os.getcwd()
        Path to the directory where the data will be saved.

    Returns
    -------
    anndata.AnnData
        Preprocessed AnnData object.
    """
    data_handler = LARRYInVitroDataset(data_dir=data_dir)
    return data_handler.adata
