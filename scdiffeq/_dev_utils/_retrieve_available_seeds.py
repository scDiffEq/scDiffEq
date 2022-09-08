import glob
import numpy as np
import pandas as pd

def _retrieve_available_seeds(model_name, return_full_dict=False):

    base_path = (
        "/home/mvinyard/benchmark/outs/{}/lightning_logs/version_*/training_summary.txt"
    )

    glob_path = base_path.format(model_name)

    available_seeds = {}
    for path in glob.glob(glob_path):
        seed = int(path.split("version_")[-1].split("/")[0])
        df = pd.read_csv(path, sep="\t", header=None)
        if "train_end" in df[0].values:
            available_seeds[seed] = df
        else:
            continue

    if return_full_dict:
        return available_seeds
    else:
        return np.sort(list(available_seeds.keys()))