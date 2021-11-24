
import torch

def _mean_square_error(y, pred_y):
    return torch.mean(torch.square(y - pred_y))