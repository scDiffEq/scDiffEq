
import torch
from geomloss import SamplesLoss

sinkhorn_loss = SamplesLoss(
    loss="sinkhorn",
    p=2,
    blur=0.05,
    reach=None,
    diameter=None,
    scaling=0.5,
    truncate=5,
    cost=None,
    kernel=None,
    cluster_scale=None,
    debias=True,
    potentials=False,
    verbose=False,
    backend="auto",
)
    
# def _MSE(predicted, truth):
    
#     mse = torch.mean(torch.square(predicted-truth))
    
#     return mse

# def _calculate_loss(predicted, truth, function="sinkhorn"):
    
#     if function == "sinkhorn":
        
#         loss = sinkhorn_loss(predicted, truth)
        
#     elif function == "MSE":
        
#         loss = _MSE(predicted, truth)
        
#     else:
#         print("Choose a loss function.")
