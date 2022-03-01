
import pandas as pd

from ._plot_smoothed_loss import _plot_smoothed_loss
from ._read_experiment import _read_log, _load_best_model
from ._read_inputs import _read_inputs

from ._simulate_trajectory import _simulate_predictions, _isolate_clonal_data
from ._plot_simulation import _plot_predicted

class _Interpolate:
    
    def __init__(self):
        
        """"""
        
        self.device = 'cpu'
        
    def read_inputs(self, adata_path, umap_path):
        
        self._adata_path = adata_path
        self._umap_path = umap_path
        self.adata, self.umap = _read_inputs(self._adata_path, self._umap_path)
        
        print(self.adata, "\n")
        self.adata_clonal = _isolate_clonal_data(self.adata)
        print(self.adata_clonal, "\n")

    def read_experiment(self, path):
        
        self.path = path
        self.df, self.log_path = _read_log(path)
        self.diffeq = _load_best_model(self.df, self.path)
        
    def plot_training(self, smoothing_window=20, save=False):
        
        self.smoothing_window = smoothing_window
        self.save = save
        
        _plot_smoothed_loss(self.df, smoothing_window, save)
        
        
    def simulate(self, n_simulations=5, batch_size=1, idx=False):
        
        """Simulate predictions using a trained model."""
        
        self.pred, self.lineage_idx = _simulate_predictions(self.adata, 
                                                    self.umap, 
                                                    self.diffeq, 
                                                    self.device,
                                                    n_simulations, 
                                                    batch_size, 
                                                    idx)
        
        self.fig = _plot_predicted(self.adata, self.adata_clonal, self.umap, self.pred, self.lineage_idx)
        
        return self.fig
        