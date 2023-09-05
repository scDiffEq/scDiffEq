

# -- import packages: ----------------------------------------------------------
import pathlib
import os
from scanpy import read
import anndata
import ABCParse


# -- set typing: ---------------------------------------------------------------
from typing import Union


# -- Controller class: ---------------------------------------------------------
class PancreaticEndocrinogenesisDataset(ABCParse.ABCParse):
    """
    Pancreatic endocrinogenesis:

    *   Downloading / io taken from scvelo
    *   Data from `Bastidas-Ponce et al. (2019) <https://doi.org/10.1242/dev.173849>`__.
    *   Pancreatic epithelial and Ngn3-Venus fusion (NVF) cells during secondary transition
        with transcriptome profiles sampled from embryonic day 15.5.
    *   Endocrine cells are derived from endocrine progenitors located in the pancreatic
        epithelium. Endocrine commitment terminates in four major fates: glucagon- producing
        α-cells, insulin-producing β-cells, somatostatin-producing δ-cells and
        ghrelin-producing ε-cells.
    """

    _URL_DATADIR = "https://github.com/theislab/scvelo_notebooks/raw/master/"

    def __init__(
        self, fpath: Union[str, pathlib.Path] = "data/Pancreas/endocrinogenesis_day15.h5ad"
    ) -> None:
        """
        Parameters:
        -----------
        fpath
            Path where to save dataset and read it from.

        Returns:
        --------
        None
        """

        self.__parse__(locals(), public=[None])

    @property
    def url(self):
        return os.path.join(self._URL_DATADIR, self._fpath)

    @property
    def _PATH_EXISTS(self):
        return os.path.exists(self._fpath) 

    @property
    def _WRITE(self):
        return (not self._PATH_EXISTS) and (self._write_h5ad)

    def __call__(self, write_h5ad: bool = True) -> anndata.AnnData:
        """
        Parameters
        ----------
        write_h5ad: bool, default = True
        
        Returns
        -------
        adata: anndata.AnnData
        """
        
        self.__update__(locals(), private = [None])

        adata = read(self._fpath, backup_url=self.url, sparse=True, cache=True)
        adata.var_names_make_unique()
        if self._WRITE:
            adata.write_h5ad(self._fpath)

        return adata

# -- API-facing function: ------------------------------------------------------
def pancreas(
    fpath: Union[pathlib.Path, str] = "data/Pancreas/endocrinogenesis_day15.h5ad",
    write_h5ad: bool = True,
    *args,
    **kwargs,
) -> anndata.AnnData:

    """
    Parameters
    ----------
    fpath: Union[pathlib.Path, str], default = "data/Pancreas/endocrinogenesis_day15.h5ad"
            Path where to save dataset and/or subsequently read it from.
            
    write_h5ad: bool, default = True
        If True and the path does not exists, the file will be written to disk.

    Returns
    -------
    adata: anndata.AnnData
    """

    data = PancreaticEndocrinogenesisDataset(fpath = fpath)
    return data(write_h5ad = write_h5ad)
