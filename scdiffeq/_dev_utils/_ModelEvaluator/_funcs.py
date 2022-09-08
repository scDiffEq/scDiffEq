import os
import pydk
import pandas as pd
import matplotlib.pyplot as plt
import vinplots


def _mk_metrics_plot():
    fig = vinplots.Plot()
    fig.construct(nplots=1, ncols=1, figsize=1.4)
    fig.modify_spines(ax="all", spines_to_delete=["top", "right"])
    ax = fig.AxesDict[0][0]
    ax.grid(zorder=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Wasserstein Distance")

    return fig, ax


def _mk_metrics_csv_path(model_path, seed):
    base_path = os.path.join(model_path, "lightning_logs/version_{}".format(seed))
    return os.path.join(base_path, "metrics.csv")


def _plot_training_metrics(model_path, seed=0, smooth=0):

    fig, ax = _mk_metrics_plot()
    metrics_path = _mk_metrics_csv_path(model_path, seed)
    metrics_df = pd.read_csv(metrics_path).groupby("epoch").mean(0).dropna()
    stdev = (
        pd.read_csv(metrics_path)
        .groupby("epoch")
        .std(0)
        .dropna()
        .rename({"train_loss_d6": "train_loss_std"}, axis=1)["train_loss_std"]
    )

    train_loss = metrics_df["train_loss_d6"]
    val_loss = metrics_df["val_loss_d6"]
    
    ceil = metrics_df[['train_loss_d6', 'val_loss_d6']].values.max()

    if smooth:
        train_loss = pydk.smooth(train_loss, smooth, smooth)
        val_loss = pydk.smooth(val_loss, smooth, smooth)

    ax.set_ylim(0, ceil)
    ax.plot(train_loss, label="Training Loss", zorder=3, lw=2, alpha=1, color="#540d6e")
    ax.plot(val_loss, label="Validation Loss", zorder=4, lw=3, alpha=1, color="#ffba49")
    ax.fill_between(
        metrics_df.index,
        train_loss - stdev,
        train_loss + stdev,
        alpha=0.5,
        color="#540d6e",
        zorder=2,
    )
    ax.set_title(model_path)

    try:
        train_summary = _read_training_summary(model_path, seed=seed)
        x = train_summary["best_epoch"]
        y = train_summary["best_score"]
        ax.scatter(x, y, c="r", s=60, zorder=6, label="best epoch: {:.2f}".format(y))
        ax.legend(edgecolor="w")
        return [metrics_df, train_summary]
    except:
        ax.legend(edgecolor="w")
        return metrics_df
    
#### -------------------------------------------------------------------------------- ####
    
def _seed_summary_path(model_path, seed):

    local_summary_path = "lightning_logs/version_{}/training_summary.txt".format(seed)
    return os.path.join(model_path, local_summary_path)


def _read_training_summary(model_path, seed):

    training_summary_path = _seed_summary_path(model_path, seed)

    f = open(training_summary_path, "r")
    summary_dict = {}
    for line in f.readlines():
        k, v = line.strip("\n").split("\t")
        summary_dict[k] = v

    return summary_dict


def _add_run_end_metrics(summary_dict):

    try:
        train_time = float(summary_dict["train_end"]) - float(
            summary_dict["train_start"]
        )
        summary_dict["epoch"] = best_epoch
        summary_dict["train_time"] = train_time
    except:
        pass

    return summary_dict


def _training_summary(model_path, seed):

    summary_dict = _read_training_summary(model_path, seed)
    summary_dict = _add_run_end_metrics(summary_dict)
    training_summary = pd.DataFrame.from_dict(
        summary_dict, orient="index", columns=["metric"]
    )

    return pd.DataFrame.from_dict(summary_dict, orient="index")


def _best_epoch_int(best_epoch_path):
    return int(os.path.basename(best_epoch_path).split("=")[1].split("-")[0])


def _get_best_epoch(summary):

    try:
        best_epoch_path = summary.loc["best_model_ckpt"].values[0]
        best_epoch = _best_epoch_int(best_epoch_path)
        return best_epoch, best_epoch_path
    except:
        return None, None
    
#### ------------------------------------------------------------------------ ####
    
import ast
import torch
from neural_diffeqs import neural_diffeq
from collections import OrderedDict


def _filter_empty_str_lines(lines):
    [lines.remove("") for i in range(lines.count(""))]


def _get_model_dict(lines):
    model_dict = {}
    for l in lines:
        if l.startswith(tuple(("mu", "sigma"))):
            name = l.split("=")
            key = name[0].strip(" ").strip(",")
            val = name[1].strip(" ").strip(",")
            if key != val:
                if "dropout" in key:
                    model_dict[key] = int(val)
                elif "activation" in key:
                    model_dict[key] = getattr(
                        torch.nn, val.strip("torch.nn").strip("()")
                    )()
                else:
                    model_dict[key] = ast.literal_eval(val)
    return model_dict


class File:
    def __init__(self, path=None):
        self._file_dict = {}
        self._path = path
        self._f = open(self._path)
        self._readlines = self._f.readlines()
        self._f.close()
        self._lines = []
        for line in self._readlines:
            self._lines.append(line.strip("\n"))

        _filter_empty_str_lines(self._lines)

    def filter_startswith(self, skip_keys=("#", "import", "from", "for")):
        """
        clean up the available lines, skipping those that you know you do not want

        Even though we could fetch the lines we actually want based on keywords like "mu"
        and "sigma", I like including this step separately because I am anticipating future
        debugging of this perhaps fragile function.
        """

        lines = []
        for line in self._lines:
            if not line.startswith(tuple(skip_keys)):
                lines.append(line.strip(" "))
        self._lines = lines

    def model_dict(self):
        """
        builds a dictionary of the arguments used to construct the model.

        These are meant to be passed as **kwargs to the neural_diffeq package.
        """
        return _get_model_dict(self._lines)


def _reconstruct_model_kwargs_from_script(path):
    f = File(path)
    f.filter_startswith()
    return f.model_dict()


def _rename_state_dict_keys(state_dict, prefix="func."):

    revised_dict = {}
    for key in state_dict.keys():
        revised_dict[key.strip(prefix)] = state_dict[key]
    return OrderedDict(revised_dict)

# from ..._models import build_custom

def _load_ckpt_state(adata, src_path, ckpt_path, alpha=0.5, use_gpus=None):
    
    """returns a lightning model"""
    
    
    if use_gpus != None:
        device = "cuda:{}".format(min(use_gpus))
    else:
        device = "cpu"
        
    model_kwargs = _reconstruct_model_kwargs_from_script(src_path)
    func = neural_diffeq(**model_kwargs)
    print("loading to device: {}".format(device))
    state = torch.load(ckpt_path, map_location=device)
#     state_dict = _rename_state_dict_keys(state["state_dict"])
    
    model = build_custom(adata, func, alpha=alpha, use_gpus=use_gpus)
    model.load_state_dict(state['state_dict'])
    
    return model

#### ------------------------------------------------------------------------ ####