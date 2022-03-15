# _download_preprocessed_data.py

import glob
import licorice
import os
import pydk


def _list_downloaded_files(destination_path, verbose, after_download=False):

    """"""

    files = glob.glob(destination_path + "/*")
    n_files = len(files)
    if n_files == 0:
        files = glob.glob(destination_path)
        msg = "Data is being downloaded to:"
    else:
        if not after_download:
            msg = "{} files have been previously downloaded to:".format(n_files)
        else:
            msg = "{} files downloaded to:".format(n_files)
    msg = licorice.font_format(
        msg, ["BOLD", "BLUE"]
    )
    if n_files > 0:
        if verbose:
            print(msg)
            for file in files:
                print("  {}".format(file))
    return files


def _download_preprocessed_anndata_from_GCP(
    destination_path="./scdiffeq_data/Weinreb2020_preprocessed/",
    bucket_path="scdiffeq-data/Weinreb2020/preprocessed_adata/*",
    force=False,
    verbose=True,
):

    """

    Parameters:
    -----------
    destination_path
        destination path for downloaded files.
        default: './scdiffeq_data/'
        type: str

    bucket_path
        path to stored data in GCP.
        default: 'scdiffeq-data/Weinreb2020/preprocessed_adata/*'
        type: str

    force
        toggle force-redownload of files from GCP.
        default: False
        type: bool

    verbose
        toggle messaging
        default: True
        type: bool
        
        
    Returns:
    --------
    downloaded_files
        list of downloaded files
        type: list(str)
    
    Notes:
    ------
    
    """

    gcp_command = pydk.gcp(destination_path, bucket_path, command="cp")
    n_files_downloaded = len(glob.glob(destination_path + "/*"))
    
    if n_files_downloaded >= 4:
        download_complete = True
    else:
        download_complete = False    

    if not download_complete or force:
        if force:
            print(licorice.font_format("\nForcing re-download...", ["BOLD", "RED"]))
        print("Downloading...")
        os.system(gcp_command)
    
    downloaded_files = _list_downloaded_files(destination_path, verbose)
    
    return downloaded_files