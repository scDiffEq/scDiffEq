
from . import _data_loader_utils as utils

import os
import scanpy as sc
import scprep
import tempfile

# EMT-specific URL to figshare
URL = "https://ndownloader.figshare.com/files/26712293"

@utils.loader
def load_EMT_simulation(save_local=False):
    """
    Download EMT simulation data from Figshare.
    
    Parameters:
    -----------
    
    save_local (default: False)
        Accepted: string, path to an optional local save destination for the EMT dataset. 
    
    
    """

    with tempfile.TemporaryDirectory() as tempdir:
        filepath = os.path.join(tempdir, "EMT_simulation.h5ad")
        scprep.io.download.download_url(URL, filepath)
        adata = sc.read(filepath)
        print(adata)
        
    if save_local != False:
        adata.write(str(save_local) + "EMT_simulation.h5ad")
        
    return adata