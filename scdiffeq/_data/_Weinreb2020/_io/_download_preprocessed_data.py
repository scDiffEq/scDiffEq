# _download_preprocessed_data.py

import glob
import licorice
import os
import pydk


def _list_downloaded_files(destination_path, verbose):

    """"""

    files = glob.glob(destination_path + "/*/*")
    msg = licorice.font_format(
        "Data has previously been downloaded to:", ["BOLD", "BLUE"]
    )
    if len(files) > 0:
        if verbose:
            print(msg)
            for file in files:
                print("  {}".format(file))
    return files


def _download_preprocessed_anndata_from_GCP(
    destination_path="./scdiffeq_data/",
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
    destination_path_exists = os.path.exists(destination_path)
    downloaded_files = _list_downloaded_files(destination_path, verbose)

    if not destination_path_exists or force:
        if force:
            print(licorice.font_format("\nForcing re-download...", ["BOLD", "RED"]))
        os.system(gcp_command)

    return downloaded_files