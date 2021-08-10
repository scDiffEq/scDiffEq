
import shutil
import tempfile
import decorator
import hashlib
import logging
import os
import atexit
import anndata as a
import scanpy as sc


"""
Data Loader Utils
--------------------
Author: Michael E. Vinyard
https://github.com/mvinyard
"""

log = logging.getLogger("scdiffeq")


def _make_tempdir():
    tempdir = os.path.join(tempfile.gettempdir(), "scdiffeq_cache")
    try:
        os.mkdir(tempdir)
        log.debug("Created data cache directory")
    except OSError:
        log.debug("Data cache directory exists")
    return tempdir


TEMPDIR = _make_tempdir()


def _cleanup():
    if os.path.isdir(TEMPDIR):
        try:
            shutil.rmtree(TEMPDIR)
            log.debug("Removed data cache directory")
        except FileNotFoundError:
            log.debug("Data cache directory does not exist, cannot remove")
        except PermissionError:
            log.debug("Missing permissions to remove data cache directory")


atexit.register(_cleanup)


def no_cleanup():
    """Don't delete temporary data files on exit."""
    atexit.unregister(_cleanup)


log = logging.getLogger("scdiffeq")


def _func_to_bytes(func):
    return bytes(".".join([func.__module__, func.__name__]), encoding="utf-8")


def _obj_to_bytes(obj):
    return bytes(str(obj), encoding="utf-8")


def _hash_function(func, *args, **kwargs):
    hash = hashlib.sha256()
    hash.update(_func_to_bytes(func))
    hash.update(_obj_to_bytes(args))
    hash.update(_obj_to_bytes(kwargs))
    return hash.hexdigest()


def _cache_path(func, *args, **kwargs):
    if hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    filename = "scdiffeq_{}.h5ad".format(_hash_function(func, *args, **kwargs))
    return os.path.join(TEMPDIR, filename)


@decorator.decorator
def loader(func, *args, **kwargs):
    """
    Decorate a data loader function.
    Taken from SingleCellOpenProblems (https://github.com/SingleCellOpenProblems)
    """
    filepath = _cache_path(func, *args, **kwargs)
    if os.path.isfile(filepath):
        log.debug(
            "Loading cached {}({}, {}) dataset".format(func.__name__, args, kwargs)
        )
        adata = a.read_h5ad(filepath)
        adata.__from_cache__ = True
        return adata
    else:
        log.debug("Downloading {}({}, {}) dataset".format(func.__name__, args, kwargs))
        adata = func(*args, **kwargs)
        adata.__from_cache__ = False
        try:
            os.mkdir(TEMPDIR)
        except OSError:
            pass
        adata.write_h5ad(filepath)
        return adata


def load_from_GCP_bucket(gcloud_path, save_destination="./", keep_local_copy=True):

    """
    Given a path to an object in a google bucket, this function downloads the
    file and returns the file path to be passed onto a data reading function.
    
    Parameters:
    -----------
    gcloud_path
        Path to google bucket item (i.e., gs://some/data)
    
    save_destination
        Where the saved filed should go. 
        
    Returns:
    --------
    file
        path to a file that was just downlaoded
    """

    file = "".join([save_destination, os.path.basename(gcloud_path)])

    if os.path.exists(file) == True:
        pass
    else:
        gsutil_command = "gsutil -m cp -r"
        gcloud_download_command = " ".join(
            [gsutil_command, gcloud_path, save_destination]
        )
        os.system(gcloud_download_command)

    return file


def load_adata_from_GCP_bucket(
    gcloud_path, save_destination="./", keep_local_copy=True
):

    """
    Loads AnnData from a Google Bucket location. Uses the general function, load_from_GCP_bucket which can grab any file from a bucket and reads it as AnnData.
    
    Parameters:
    -----------
    gcloud_path
        Path to google bucket item (i.e., gs://some/data)
    
    save_destination
        Where the saved filed should go. 
    
    keep_local_copy
        If False, the file will be erased from disk once loaded into memory. 
    
    Returns:
    --------
    adata
        AnnData object
    
    """

    downloaded_file = load_from_GCP_bucket(
        gcloud_path, save_destination, keep_local_copy
    )
    adata = a.read_h5ad(downloaded_file)
    
    if keep_local_copy == False:
        os.remove(downloaded_file)

    return adata