# _read_downloaded_data.py

import adata_sdk
import anndata as a
import numpy as np
import pydk


def _parse_organize_available_datatypes(files):

    """"""

    data = {}

    for file_specification in ["h5ad", "pca", "umap"]:
        data[file_specification] = np.array(files)[
            [file.endswith(".{}".format(file_specification)) for file in files]
        ][0]

    return data


def _load_CytoTRACE_annotations(adata, path, use_cols=False):

    """"""

    adata_sdk.parse_to_obs(adata, path, use_cols)


def _prepare_anndata(files, CytoTRACE_path):

    """"""

    data = _parse_organize_available_datatypes(files)
    adata = a.read_h5ad(data["h5ad"])
    adata.uns["pca"] = pydk.load_pickled(data["pca"])
    adata.uns["umap"] = pydk.load_pickled(data["umap"])

    _load_CytoTRACE_annotations(adata, CytoTRACE_path, use_cols=False)

    return adata


def _read_downloaded_data(downloaded_files, CytoTRACE_path, verbose=False):

    return _prepare_anndata(downloaded_files, CytoTRACE_path)