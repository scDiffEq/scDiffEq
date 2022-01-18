
import os
import pandas as pd
import scdiffeq as sdq
import torch


def _read_log(out_path):
    
    log_path = os.path.join(out_path, "status.log")
    
    df = pd.read_csv(log_path, sep="\t")
    
    return df, log_path


def _load_best_model(df, path):
    
    """"""

    best_model = df.training_loss.argmin()
    best_model_path = os.path.join(path, "best.model.epoch_{}".format(best_model))

    diffeq = sdq.scDiffEq(in_dim=50, out_dim=50, nodes=50, layers=2, device="cpu")
    diffeq.network_model.load_state_dict(torch.load(best_model_path, map_location=torch.device("cpu")))
    
    return diffeq

