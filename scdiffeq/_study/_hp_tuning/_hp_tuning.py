

import glob
import numpy as np
import os
import pandas as pd

from ._plot_training_loss_parameter_tuning import _plot_training_loss_parameter_tuning
from ._load_best_models import _load_best_models
from ._load_best_models import _get_available_loaded_best_models
from ._load_preprocessed_Weinreb2020_data import _load_preprocessed_Weinreb2020_data
from .._utilities._get_best_model import _get_best_model
from ..._utilities._count_network_parameters import _count_parameters_all_models
from ._ensemble_best_training_results import _ensemble_best_training_results

def _return_status_log_paths(outpath):

    return glob.glob(os.path.join(outpath + "*/status.log"))


def _create_StatusLogDict(
    StatusLogDict, filename, columns=["seeds", "layers", "nodes"]
):

    """"""

    StatusLogDict[filename] = {}
    for col in columns:
        StatusLogDict[filename][col] = []

    return StatusLogDict


def _parse_status_filename(StatusLogDict, file):

    filename = os.path.basename(os.path.dirname(file))
    StatusLogDict = _create_StatusLogDict(
        StatusLogDict, filename, columns=["layers", "nodes", "seed"]
    )
    StatusLogDict[filename]["layers"] = filename.split("_")[2].split(".")[1]
    StatusLogDict[filename]["nodes"] = filename.split("_")[1].split(".")[1]
    StatusLogDict[filename]["seed"] = filename.split("_")[1].split(".")[0]

    return StatusLogDict


def _load_hyperparamter_table(outpath):

    """"""

    StatusLogDict = {}
    status_files = _return_status_log_paths(outpath)
    
    for file in status_files:
        StatusLogDict = _parse_status_filename(StatusLogDict, file)
    print(StatusLogDict)
    return (
        pd.DataFrame.from_dict(StatusLogDict)
        .astype(int)
        .T.sort_values(["layers", "nodes", "seed"])
        .reset_index(drop=True)
    )


def _get_single_condition_logs(path, layers, nodes):

    return glob.glob(
        path + "*.{}_nodes*{}_layers*/status.log".format(nodes, layers)
    )  # len = n_seeds


def _make_df(HPTuningLog_1Condition):

    """"""

    df_cols = np.sort(list(HPTuningLog_1Condition.keys())).astype(int)

    tuning_df = pd.DataFrame(HPTuningLog_1Condition)[df_cols]
    tuning_df["mean"] = tuning_df.mean(axis=1)
    tuning_df["std"] = tuning_df.std(axis=1)
    tuning_df["lower"] = tuning_df["mean"] - tuning_df["std"]
    tuning_df["upper"] = tuning_df["mean"] + tuning_df["std"]

    return tuning_df


def _get_HyperParameterTuningLogs(hp_table_df, outpath):

    HPTuningLogs = {}
    HPTuningDataFrames = {}
    for group, rows in hp_table_df.groupby(["layers", "nodes"]):
        layers, nodes = group[0], group[1]
        log_paths = _get_single_condition_logs(outpath, layers, nodes)
        HPTuningLogs["layers_{}.nodes_{}".format(layers, nodes)] = {}
        for path in log_paths:
            seed = int(path.split("seed_")[1].split(".")[0])
            df_tmp = pd.read_csv(path, sep="\t")
            HPTuningLogs["layers_{}.nodes_{}".format(layers, nodes)][
                seed
            ] = df_tmp[df_tmp.columns[-1]]

    return HPTuningLogs

def _make_HPTuningDict(hp_table_df, outpath):

    HPTuning_dfs = {}
    HPTuningLogs = _get_HyperParameterTuningLogs(hp_table_df, outpath)

    for hp_key, hp_values in HPTuningLogs.items():
        HPTuning_dfs[hp_key] = _make_df(hp_values)

    return HPTuning_dfs


def _fetch_HyperParameterTuning(outpath):

    hp_table_df = _load_hyperparamter_table(outpath)
    results_df_dict = _make_HPTuningDict(hp_table_df, outpath)

    return hp_table_df, results_df_dict

def _annotate_tuning_with_best_model_epoch(outpath, hp_table):

    best_model_epoch = []
    best_model_paths = []

    for row in range(len(hp_table)):
        df_row = hp_table.iloc[row]
        _path = os.path.join(outpath, "seed_{}.{}_nodes.{}_layers.cuda_*".format(
            df_row.seed, df_row.nodes, df_row.layers
        ))
        try:
            best_model = _get_best_model(_path)
            best_model_epoch.append(best_model.epoch)
            best_model_paths.append(best_model.path)
        except:
            best_model_epoch.append(None)
            best_model_paths.append(None)

    hp_table["best_model_epoch"] = best_model_epoch
    hp_table["best_model_path"] = best_model_paths
    
    return hp_table

class _HyperParameterTuningMonitor:
    def __init__(self, outpath, datapath, in_dim=50, out_dim=50, device ="cpu"):

        """"""

        self._outpath = outpath
        self._datapath = datapath
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._device = device
        
        
    def load_data(self):
        
        self.data = _load_preprocessed_Weinreb2020_data(self._datapath)
        
    def load_training(self):

        """"""
        self._hp_table, self._results_dict = _fetch_HyperParameterTuning(self._outpath)
        self._hp_table = _annotate_tuning_with_best_model_epoch(self._outpath, self._hp_table)
        self._best_available_models = _load_best_models(self._hp_table, self._in_dim, self._out_dim, self._device)
    
    def load_best_models(self):
        
        self.best_models = _get_available_loaded_best_models(self._best_available_models)
        param_df = _count_parameters_all_models(self.best_models)

        for column in ["layers", "nodes"]:
            param_df[column] = param_df[column].astype(int)
        self._hp_table = self._hp_table.merge(param_df, on=["layers", "nodes"])
        
    def plot(self, nplots=False, ncols=2):
        
        if not nplots:
            nplots = len(self._hp_table)
        
        _plot_training_loss_parameter_tuning(self._hp_table, self._results_dict, nplots, ncols)
        
    def ensemble_best_training(self, return_dict=True):
        
        self._ensemble_df = _ensemble_best_training_results(self._results_dict, return_dict)
            
def _load_hp_tuning_results(
    results_path,
    adata_path,
    plot=True,
    nplots=False,
    ncols=2,
    in_dim=50,
    out_dim=50,
    device="cpu",
):

    """

    Parameters:
    -----------
    results_path
        type: str

    adata_path
        type: str

    plot
        type: bool
        default: True

    nplots
        type: int or bool
        default: False

    ncols
        type: int
        default: 2

    in_dim
        type: int
        default: 50

    out_dim
        type: int
        default: 50

    device
        type: str
        default: "cpu"

    Returns:
    --------
    HP_tuning
        Custom class for hyper parameter tuning module.

    Notes:
    ------
    (1) Easily accessible:
        - adata = HP_tuning.data.adata
        - best_models = HP_tuning.best_models
    """

    HP_tuning = _HyperParameterTuningMonitor(results_path, adata_path)
    HP_tuning.load_training()
    if plot:
        HP_tuning.plot(nplots, ncols)
    HP_tuning.load_data()
    HP_tuning.load_best_models()
    HP_tuning.ensemble_best_training()

    return HP_tuning