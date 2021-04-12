import pickle as pk
import numpy as np
import pandas as pd
import anndata as a

def save_pca_as_pkl(adata):

    """"""

    pca_path = str(adata.uns["run_id"]) + "_pca.pkl"

    pca = adata.uns["pca"]
    pk.dump(pca, open(pca_path, "wb"))
    adata.uns["pca"] = pca_path


def fix_adata_for_writing(adata):

    """"""

    save_pca_as_pkl(adata)
    adata.uns["odefunc"] = None
    adata.uns["optimizer"] = None
    adata.uns["loss_meter"] = None
    adata.uns["time_meter"] = None
    adata.uns["device"] = None
    adata.uns["data_split_keys"] = None
    adata.uns["latest_training_predictions"] = None
    adata.uns["latest_test_predictions"] = None
    adata.uns["latest_validation_predictions"] = None
    adata.uns["latest_training_true_y"] = None
    adata.uns["latest_test_true_y"] = None
    adata.uns["latest_validation_true_y"] = None
    try:
        adata.uns["epoch_counter"] = np.array(adata.uns["epoch_counter"])
    except:
        pass
    try:
        adata.uns["training_loss"] = np.array(adata.uns["training_loss"]).astype(str)
    except:
        pass
    try:
        adata.uns["validation_loss"] = np.array(adata.uns["validation_loss"]).astype(str)
    except:
        pass

def write_h5ad(adata, path=None):
    
    temp_func = adata.uns["odefunc"]
    temp_optim = adata.uns["optimizer"]
    temp_loss_meter = adata.uns["loss_meter"]
    temp_time_meter = adata.uns["time_meter"]
    temp_device = adata.uns["device"]
    temp_l_train_p = adata.uns["latest_training_predictions"]
    temp_l_val_p = adata.uns["latest_validation_predictions"]
    temp_l_train_t = adata.uns["latest_training_true_y"]
    temp_l_val_t = adata.uns["latest_validation_true_y"]
    
    try:
        temp_l_test_p = adata.uns["latest_test_predictions"]
        temp_l_test_t = adata.uns["latest_test_true_y"]
    except:
        pass
    
    if path == None:
        path = str(adata.uns["run_id"]) + ".h5ad"

    fix_adata_for_writing(adata)
    adata.write_h5ad(path)
    
    adata.uns["odefunc"] = temp_func
    adata.uns["optimizer"] = temp_optim
    adata.uns["loss_meter"] = temp_loss_meter
    adata.uns["time_meter"] = temp_time_meter
    adata.uns["device"] = temp_device
    adata.uns["latest_training_predictions"] = temp_l_train_p
    adata.uns["latest_validation_predictions"] = temp_l_val_p
    adata.uns["latest_training_true_y"] = temp_l_train_t
    adata.uns["latest_validation_true_y"] = temp_l_val_t
    
    try:
        adata.uns["latest_test_predictions"] = temp_l_test_p
        adata.uns["latest_test_true_y"] = temp_l_test_t
    except:
        pass
    
    print("AnnData object saved to", str(path))