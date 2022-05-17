
import numpy as np
import os


def _load_saved_true_bias_scores(
    relative_path="_precalculated_scores/true_fate_biases.335_cells.test_set.npy",
):
    path = os.path.dirname(os.path.dirname(__file__))
    
    return np.load(os.path.join(path, relative_path))
