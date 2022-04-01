
import torch
import annoyance
from collections import Counter
import numpy as np

def _get_X_pred_evaluate(X_pred_epoch, t_evaluate = [4, 6], time_dict = {2:0, 4:1, 6:2}):
    
    X_evaluate = []
    
    for t in t_evaluate:
        X_evaluate.append(X_pred_epoch[time_dict[t]])
    
    return torch.stack(X_evaluate)

def _break_potential_label_count_tie(nn, label, count):
    
    _, count_2 = nn[1]
    if count == count_2:
        return "Other"
    else:
        return label
    
    
def _declare_most_populous_fate(nn):
    
    label, count = nn[0]
    
    if len(nn) > 1:
        label = _break_potential_label_count_tie(nn, label, count)
    
    return label, count

def _count_nearest_neighbor_cell_labels(X_cell_evaluate, n_neighbors, AnnoyModel, n_most_common=2):
    
    annoy_idx = AnnoyModel._annoy_idx
    cell_idx  = AnnoyModel._cell_idx
    
    X_pred_labels = []
    
    for cell_vector in X_cell_evaluate:
        nn_pred_idx = annoy_idx.get_nns_by_vector(cell_vector, n_neighbors)
        _nn = cell_idx[nn_pred_idx]
        _nn = Counter(_nn).most_common(n_most_common)
        label, count = _declare_most_populous_fate(_nn)
        X_pred_labels.append(label)
    
    return Counter(X_pred_labels)

def _calculate_cell_fate_bias(X_pred_epoch, annoy_path, t_evaluate = [4, 6], n_neighbors=20, dim=50):
    
    """
    
    Returns:
    --------
    CellFateBiasDict
    """
    
    n_cells = X_pred_epoch.shape[-2]
    
    AnnoyModel = annoyance.annoy()
    AnnoyModel.load(annoy_path)
    
    X_evaluate = _get_X_pred_evaluate(X_pred_epoch, t_evaluate, time_dict = {2:0, 4:1, 6:2})

    scores = []
    mask = []
    
    for cell_i in range(n_cells):
        X_cell_evaluate = X_evaluate[:, :, cell_i, :].reshape(-1, dim)
        X_pred_labels = _count_nearest_neighbor_cell_labels(X_cell_evaluate, n_neighbors=n_neighbors, AnnoyModel=AnnoyModel)        
        
        num_neu = X_pred_labels["Neutrophil"] + 1  # use pseudocounts for scoring
        num_total = X_pred_labels["Neutrophil"] + X_pred_labels["Monocyte"] + 2
        score = num_neu / num_total
        scores.append(score)
        num_total = X_pred_labels["Neutrophil"] + X_pred_labels["Monocyte"]
        mask.append(num_total > 0)
    
    CellFateBiasDict = {}
    CellFateBiasDict['scores'] = np.array(scores)
    CellFateBiasDict['mask'] = np.array(mask)
    CellFateBiasDict['n_masked'] = len(scores) - sum(mask)
    
    
    return CellFateBiasDict
