
from . import _data_loader_utils as utils
import warnings
warnings.filterwarnings("ignore")

def load_LARRY(save_destination="./", keep_local_copy=True):
    
    """
    This function downloads the h5ad file from google cloud and then uses anndata to return the adata object.
    
    Parameters:
    -----------
    save_destination
        Path for the downloaded file.
    
    keep_local_copy
        If False, this erases the file from the local disk once loaded into memory. 
    
    Returns:
    --------
    adata
        AnnData object of the LARRY Dataset. 
    """
    
    data_path = "gs://klein-data-science-2020/LARRY.h5ad"
    adata = utils.load_adata_from_GCP_bucket(data_path, save_destination, keep_local_copy)
    print(adata)
    
    return adata