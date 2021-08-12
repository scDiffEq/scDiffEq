
from tqdm import tqdm_notebook as tqdm_nb

from ._forward_integration_functions._forward_integrate_epoch import _forward_integrate_epoch
from ._plot_neural_ODE_training import _plot_loss

def _learn_neural_ODE(
    adata, n_epochs=100, plot_progress=False, plot_summary=True, smoothing_factor=3, visualization_frequency=5,
):

    """"""
    
    for epoch in tqdm_nb((range(1, n_epochs + 1)), desc="Training progress"):
    
#     for epoch in range(1, (n_epochs + 1)):
        _forward_integrate_epoch(adata, epoch)
        if plot_progress:
            if epoch % visualization_frequency == 0:
                _plot_loss(adata, groupsize=smoothing_factor)
    
    if plot_summary:
        _plot_loss(adata, groupsize=smoothing_factor)