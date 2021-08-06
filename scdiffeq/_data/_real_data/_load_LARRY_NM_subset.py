from vintools.utilities import pyGSUTILS
import os, glob, pickle, vintools as v, anndata as a
from .._utilities._AnnData_handlers._split_AnnData_test_train_validation import _split_test_train

gsutil = pyGSUTILS()

def _read_local_h5ad_adata(h5ad_path, adjacent_pkl=True):
    
    dirname = os.path.dirname(h5ad_path)
    pkl = glob.glob(dirname + "/*.pkl")
    
    print("Reading AnnData...")
    adata = a.read_h5ad(h5ad_path)
    
    if len(pkl) == 1:
        pkl=pkl[0]
        try:
            adata.uns["pca"] = pickle.load(open(pkl, "rb"))
            print(".pkl file with PCA information saved to adata.uns['pca']")
        except:
            print(".pkl file detected but not properly loaded.")
    elif len(pkl) > 1:
        print("More than one .pkl file detected...\n\n{}".format(pkl))
    else:
        print("Something went wrong...")
    
    return adata


def _get_LARRY_NM_subset(destination_path="./.scdiffeq_cache/", return_AnnData=False):
    
    """
    Downloads to a cache dir *once*. If already available, loads and splits into test-train. 

    Parameters:
    -----------
    [optional] destination_path
        Default: "./.scdiffeq_cache", 
        type: str

    [optional] return_AnnData
        If True, both the split data object as well as the AnnData object are returned. Otherwise, only the split data object is returned. 
        Default: False
        type: bool

    Returns:
    --------
    data or: data, AnnData

    Data structure:

    data
    │
    ├── train
    │   ├── obs
    │   ├── index
    │   ├── data
    │   ├── t
    │   └── emb
    ├── validation
    │   ├── obs
    │   ├── index
    │   ├── data
    │   ├── t
    │   └── emb
    ├── test
        ├── obs
        ├── index
        ├── data
        ├── t
        └── emb

    Where "emb" is the pca embedding in 50 dimensions. 

    Notes:
    ------

    (1) test-train at current is random / generated on the fly. 
    
    (2) to work with gsutils:
        (2a) conda install -c conda-forge google-cloud-sdk -y (or preferred method of installation)
        (2b) run in terminal: gcloud auth login
        
    (3) Download is 8.1 GB. This might take a minute or two and will display the following, minimally helpful update:

        Copying gs://scdiffeq-data/LARRY_Neutrophil_Monocyte_Subset/LARRY.NM.subset.h5ad...
        | [0/1 files][  4.7 GiB/  8.1 GiB]  58% Done  49.5 MiB/s ETA 00:01:11    

        Don't be alarmed if it doesn't updated rapidly. If you run `ls -a` in the target cache directory, you should see:

            scdiffeq_cache_.gstmp

        If download is interrupted, you might see something like the following: 

            Copying gs://scdiffeq-data/LARRY_Neutrophil_Monocyte_Subset/LARRY.NM.subset.h5ad...
            Resuming download for ./.scdiffeq_cache component 0                             
            Resuming download for ./.scdiffeq_cache component 1
            Resuming download for ./.scdiffeq_cache component 2
            Resuming download for ./.scdiffeq_cache component 3
            \ [0/1 files][  5.1 GiB/  8.1 GiB]  62% Done  52.2 MiB/s ETA 00:01:00           

    """
    
    # gsutil paths:
    _h5ad_path = (
        "scdiffeq-data/LARRY_Neutrophil_Monocyte_Subset/LARRY.NM.subset.h5ad"
    )
    _pkl_path = (
        "scdiffeq-data/LARRY_Neutrophil_Monocyte_Subset/LARRY.NM.subset.pca.pkl"
    )
    
    # download
    path_to_data_h5ad = os.path.join(destination_path, os.path.basename(_h5ad_path))
    path_to_data_pkl = os.path.join(destination_path, os.path.basename(_pkl_path))


    if os.path.exists(path_to_data_h5ad):
        print(
            "\nData already downloaded! Using cached data from {}".format(
                v.ut.format_pystring(path_to_data_h5ad, ["BOLD", "RED"])
            )
        )
    else:
        gsutil.cp(_h5ad_path, destination_path)

    if os.path.exists(path_to_data_pkl):
        print(
            "\nData already downloaded! Using cached data from {}".format(
                v.ut.format_pystring(path_to_data_pkl, ["BOLD", "RED"])
            )
        )
    else:
        gsutil.cp(_pkl_path, destination_path, verbose=True)
    
    # read_cached or newly downloaded data
    adata = _read_local_h5ad_adata(h5ad_path=path_to_data_h5ad, adjacent_pkl=True)

    # split data into test_train_validation
    data = _split_test_train(adata, time_column='cytoTIME')

    if return_AnnData:
        return data, adata
    
    return data

