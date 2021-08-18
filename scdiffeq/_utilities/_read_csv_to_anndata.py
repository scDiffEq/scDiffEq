import anndata as a
import pandas as pd
import numpy as np


def _get_obs_and_vars(adata, df, var_names):

    """Set adata.vars and adata.obs. Time column assumed to be the 0th column."""

    # time column assumed to be the 0th column
    adata.obs = pd.DataFrame(df[["time", "i2"]])

    # gives one the ability to "subset" all AnnData later
    adata.obs["all_data"] = True

    # takes all column names as gene names aside from the time column
    adata.var = pd.DataFrame(df.columns[1:-1])
    adata.var.set_index(0, drop=True, inplace=True)
    adata.var.index.rename(var_names, inplace=True)

    return adata


def _make_adata(df, var_names):

    # takes all columns after excluding the time column (0th column)
    adata = a.AnnData(np.array(df)[:, 1:-1])

    adata = _get_obs_and_vars(adata, df, var_names)

    return adata


def _read_csv_to_anndata(
    path, names=None, header=None, skiprows=1, var_names="features", usecols=None
):

    """
    Converts a CSV gene expression matrix to an AnnData object. 
    Make sure to put the time column first!
    
    A wrapper for one in a series of functions from SCANPY/AnnData, which collects and formats objects as an AnnData object, commonly referred to as `adata`. (Wolf, et al., Genome Biology, 2018). 
    
    Function receives a path to an AnnData.h5ad file and returns an AnnData object. more information available at https://anndata.readthedocs.io/en/latest/anndata.AnnData.html
    
    Parameters:
    -----------
    path (required)
        path to data
    names (optional)
        a vector of specified feature names.
        Default: None
    header (optional)
        Default: None
    skiprows (optional)
        Default: 1
    var_names (optional)
        how adata.vars will be headed and the primary descriptor of the var axis of the feature matrix.
        Default: "features"
    usecols (optional)
        if there are columns you wish to exclude, make sure to specify which columns you'd like to use here. 
        Default: None
        
    Returns:
    --------
    adata
        AnnData object
    """

    df = pd.read_csv(
        path, usecols=usecols, header=header, skiprows=skiprows, names=names
    )

    adata = _make_adata(df, var_names)

    print(adata)

    return adata
