
import pathlib
import os
import numpy as np
import anndata

from ..scdiffeq import io

class TestAnnData:
#    def __init__(self, *args, **kwargs):
#        ...

    @property
    def X(self):
        return np.array(
            [
                [0, 0, 0, 0, 4, 0, 3],
                [0, 0, 0, 4, 0, 0, 2],
                [0, 0, 4, 0, 0, 0, 1],
                [0, 4, 0, 0, 0, 0, 5],
                [4, 0, 0, 0, 0, 0, 8],
                [4, 0, 0, 0, 0, 0, 4],
            ]
        )

    @property
    def t(self):
        return np.array([0, 0, 1, 1, 2, 2])

    @property
    def cell_barcodes(self):
        return [
            "ABC123",
            "ABC456",
            "DEF123",
            "DEF456",
            "ABC789",
            "DEF789",
        ]

    @property
    def cluster(self):
        return [
            "A",
            "B",
            "C",
            "A",
            "B",
            "C",
        ]

    @property
    def adata(self):
        adata = anndata.AnnData(X=self.X) # , dtype=self.X.dtype)
        adata.obs["cell_barcode"] = self.cell_barcodes
        adata.obs["t"] = self.t
        adata.obs["cluster"] = self.cluster
        return adata

    def __call__(self, *args, **kwargs):

        return self.adata


def test_read_h5ad():

    h5ad_path = pathlib.Path("pytest_adata.h5ad")

    if not h5ad_path.exists():
        test_anndata = TestAnnData()

        adata = test_anndata()
        adata.write_h5ad(h5ad_path)

    adata = io.read_h5ad(h5ad_path)

    assert adata is not None, "AnnData object is None"
    assert hasattr(adata, "X"), "AnnData object doesn't have the main data matrix"

    os.remove("pytest_adata.h5ad")
