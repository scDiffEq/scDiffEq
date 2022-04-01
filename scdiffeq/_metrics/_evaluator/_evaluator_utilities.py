
from numba import cuda
import os
import pydk
import torch

def _release_GPU_memory():              

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)
    
def _specify_device(device):
    
    if type(device) is int:
        return "cuda:{}".format(device)
    else:
        return "cpu"

def _make_evaluation_outpath(evaluation_outpath, run_signature):

    if not evaluation_outpath:
        evaluation_outpath = "./"

    outpath = os.path.join(evaluation_outpath, "scDiffEq_evaluated", run_signature)
    pydk.mkdir_flex(outpath)
    
    return outpath

def _write_predicted_labels(X_labels, N, evaluation_outpath):
    
    for key, value in X_labels.items():
        score_save_path = os.path.join(evaluation_outpath, "epoch_{}_fate_score.pt".format(str(key)))
        ScoreDict = {}
        for _key in value.keys():
            write_key = "_".join(["epoch", str(key), str(_key)])
            ScoreDict[write_key] = value[_key]
        torch.save(ScoreDict, score_save_path)