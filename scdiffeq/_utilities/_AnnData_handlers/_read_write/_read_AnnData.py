from glob import glob
import anndata as a
import os, pickle

from .._downsample_AnnData import _downsample_AnnData
from .._split_AnnData_test_train_validation import _split_test_train

def _choose_path_from_glob_list(glob_path_list):

    """"""

    if len(glob_path_list) > 1:
        print(
            glob_path_list,
            "; please narrow your search criteria using the `label` parameter.",
        )
    else:
        glob_path_item = glob_path_list[0]

    return glob_path_item


def _read_AnnData(
    label="testing_results",
    outpath="./",
    scdiffeq_outs_dir="scdiffeq_adata",
    downsample_percent=1,
    downsample_n_trajectories=False,
    downsample_sort_on=["trajectory", "time"],
    split_data_test_train=True,
    data_split_trajectory_column='trajectory',
    data_split_proportion_training=0.6,
    data_split_proportion_validation=0.2,
    data_split_return_data_subsets=False,
    data_split_time_column='time',
    silence=False,
):

    """
    Writes h5ad and pkl file for outputs of scdiffeq method.
    Mainly written to fill the gaps of AnnData (i.e., cannot save pca result from sklearn).
    Creates output directory if needed.

    Parameters:
    -----------
    adata
        AnnData object.

    label
        experiment-specific label.

    outpath
        directory where outs_directory should be placed.

    scdiffeq_outs_dir
        scdiffeq-specific outs directory
        
    downsample_percent
        percentage of data to be retained (max: 1, min: 0; e.g.: 5% is 0.05)
        default: 1
        
    downsample_sort_on
        keys contained in adata.obs on which AnnData object should be sorted prior to downsampling
        default: ["trajectory", "time"] (useful for scdiffeq; might eventually change to make more general or if implemented in vintools.)
    
    split_data_test_train
        
        default: True
        
    data_split_trajectory_column
        
        default: 'trajectory'
        
    data_split_proportion_training
        
        default: 0.6
        
    data_split_proportion_validation
        
        default: 0.2
        
    data_split_return_data_subsets
        
        default: False
        
    data_split_time_column
        
        default: 'time'
        
    
    silence
        if True, silence prevents function from printing resulting AnnData attributes.
        default: False
    
    Returns:
    --------
    None
    """

    h5ad_search_path = _choose_path_from_glob_list(
        glob(os.path.join(outpath, scdiffeq_outs_dir, (label + "*h5ad")))
    )
    pkl_search_path = _choose_path_from_glob_list(
        glob(os.path.join(outpath, scdiffeq_outs_dir, (label + "*.pkl")))
    )

    adata = a.read_h5ad(h5ad_search_path)
    adata.uns["pca"] = pickle.load(open(pkl_search_path, "rb"))

    if downsample_percent != 1:
        print("Downsampling AnnData...")
        adata = _downsample_AnnData(
            adata, percent=downsample_percent, n_traj=downsample_n_trajectories, sort_on=downsample_sort_on, silence=True
        )
    
    if split_data_test_train:
        _split_test_train(
            adata,
            trajectory_column=data_split_trajectory_column,
            proportion_training=data_split_proportion_training,
            proportion_validation=data_split_proportion_validation,
            return_data_subsets=data_split_return_data_subsets,
            time_column=data_split_time_column,
            silent=True,
        )

    if not silence:
        print(adata)

    return adata