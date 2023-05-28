

import os
from scanpy import read
from pathlib import Path
from typing import Union


from ..core import utils


class PancreaticEndocrinogenesisDataset(utils.ABCParse):
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
        self, fpath: Union[str, Path] = "data/Pancreas/endocrinogenesis_day15.h5ad"
    ):
        """
        Parameters:
        -----------
        fpath
            Path where to save dataset and read it from.

        Returns:
        --------
        None
            type: NoneType
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
        return not self._PATH_EXISTS

    def __call__(self):
        """
        Returns:
        --------
        adata
            type: anndata.AnnData
        """

        WRITE = self._WRITE

        adata = read(self._fpath, backup_url=self.url, sparse=True, cache=True)
        adata.var_names_make_unique()
        if WRITE:
            adata.write_h5ad(data._fpath)

        return adata


def pancreas(fpath: Union[str, Path] = "data/Pancreas/endocrinogenesis_day15.h5ad"):
    data = PancreaticEndocrinogenesisDataset(fpath=fpath)
    return data()