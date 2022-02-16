import anndata as a
import glob
import licorice
import numpy as np
import os
import pickle
import pyrequisites as pyrex

from ._Weinreb2020_Figure5_Annotations import _annotate_adata_with_Weinreb2020_Fig5_predictions


def _list_downloaded_files(destination_path):
    
    """"""
    
    files = glob.glob(destination_path + "/*")
    msg = licorice.font_format("Data has previously been downloaded to:", ['BOLD', 'BLUE'])
    print(msg)
    for file in files:
        print("  {}".format(file))
    
    return files
    
def _mk_gcp_command(destination, bucket_path):
    
    destination_path = os.path.join(destination, "scdiffeq_data/Weinreb2020_preprocessed/")
    pyrex.mkdir_flex(destination_path)
    gcp_command = "gsutil -m cp -r gs://{} {}".format(bucket_path, destination_path)
    
    return gcp_command, destination_path

def _download_preprocessed_anndata(destination="./", bucket_path = "scdiffeq-data/Weinreb2020/preprocessed_adata/*", force=False):
    
    """"""
    
    gcp_command, destination_path = _mk_gcp_command(destination, bucket_path)
    if os.path.exists(destination_path):
        files = _list_downloaded_files(destination_path)
        if force:
            print(licorice.font_format("\nForcing re-download...", ['BOLD', 'RED']))
            os.system(gcp_command)
    else:
        os.system(gcp_command)
        files = glob.glob(destination_path + "/*")
        
    return destination_path, files

def _load_pickled(path):
    return pickle.load(open(path, 'rb'))

def _prepare_anndata(files, silent=False):
    
    """"""
    
    data = {}
    for file_specification in ['h5ad', 'pca', 'umap']:
        data[file_specification] = np.array(files)[[file.endswith(".{}".format(file_specification)) for file in files]][0]
    
    adata = a.read_h5ad(data['h5ad'])
    adata.uns['pca'] = _load_pickled(data['pca'])
    adata.uns['umap'] = _load_pickled(data['umap'])
    
    return adata

def _load_Weinreb2020_preprocessed(destination="./",
                                   bucket_path = "scdiffeq-data/Weinreb2020/preprocessed_adata/*", 
                                   silent=False,
                                   force=False):
    
    """
    Load the preprocessed Weinreb (2020) Science dataset as preprocessed by PRESCIENT authors. 
    
    Parameters:
    -----------
    destination
        default: "./"
        type: str
    
    bucket_path
        default: "scdiffeq-data/Weinreb2020/preprocessed_adata/*"
        type: str
    
    force
        default: False
        type: bool
    
    Returns:
    --------
    adata
        type: anndata.AnnData
    
    Notes:
    ------
    
    """
    
    destination_path, files = _download_preprocessed_anndata(destination, bucket_path, force)
    adata = _prepare_anndata(files, silent)
    adata = _annotate_adata_with_Weinreb2020_Fig5_predictions(adata)
    
    if not silent:
        print("\n{}".format(adata))
    
    return adata