
import os
import torch

from ._load_trained_model import _load_trained_model
from ._load_trained_model import _parse_model_training_path
from ._plot_model_training import _plot_training
from ._pass_test_data_to_model import _pass_test_data_to_model
from ._parse_results_functions import _return_epochs_to_evaluate
from ._calculate_cell_fate_bias import _calculate_cell_fate_bias
from ._determine_model_accuracy import _determine_model_accuracy
from ._plot_model_accuracy import _plot_model_accuracy
from ._evaluator_utilities import _make_evaluation_outpath
from ._evaluator_utilities import _write_predicted_labels

class _Evaluator:
    
    def __init__(self, path=False):
        
        """"""
        
        if path:
            _parse_model_training_path(self, path)
            
    def load(self, path=False, evaluation_outpath=False):
        
        if path:
            _parse_model_training_path(self, path)
            
        self.log_df = _load_trained_model(self._log_path, self._training_model_paths)
        self.best_epoch = self.log_df[self.log_df['best']]
        
        self._epochs_to_evaluate = _return_epochs_to_evaluate(self.log_df)
        self._evaluation_outpath = _make_evaluation_outpath(evaluation_outpath,
                                                            self._run_signature,
                                                           )        
        
    def plot_training(self, save=True, figsize=1.5):
        
        _plot_training(self.log_df,
                       savename=save,
                       outpath=self._evaluation_outpath,
                       layers=self._layers,
                       nodes=self._nodes,
                       seed=self._seed,
                       figsize=figsize,
                      )
        
    def pass_test_data_to_model(self, adata, N=2, device=0):
        
        self._adata = adata
        self._N = N
        
        self._X_pred = _pass_test_data_to_model(adata,
                                                self._epochs_to_evaluate,
                                                self._path,
                                                self._layers,
                                                self._nodes,
                                                self._seed,
                                                self._evaluation_outpath,
                                                N=self._N,
                                                device=device,
                                               )
    
    def calculate_cell_fate_bias(self, annoy_path, t_evaluate=[4, 6], n_neighbors=20, dim=50):
        
        self._X_labels = {}
        
        for epoch, X_pred_epoch in self._X_pred.items():
            scores, mask, n_masked = _calculate_cell_fate_bias(X_pred_epoch,
                                                               annoy_path=annoy_path,
                                                               t_evaluate=t_evaluate ,
                                                               n_neighbors=n_neighbors,
                                                               dim=dim,
                                                              )
            
            self._X_labels[epoch] = {}
            self._X_labels[epoch]["scores"] = scores
            self._X_labels[epoch]["mask"] = mask
            self._X_labels[epoch]["n_masked"] = n_masked
            
        _write_predicted_labels(self._X_labels, self._N, self._evaluation_outpath)
            
    def determine_model_accuracy(self,
                                 adata=False,
                                 plot=True,
                                 plot_markersize=140,
                                 plot_approx_prescient=[0.43, 0.72],
                                 plot_model_color="#0a9396",
                                 plot_y_labels=["Pearson's Rho", "AUROC"],
                                 plot_savename=False,
                                 plot_width_ratios=[1, 0.2, 1],
                                 plot_wspace=0.2,
                                 figsize_width=0.5,
                                 figsize_height=1,
                                 figsize=False,
                                ):
        
        if adata:
            self._adata = adata
            
        self._accuracy_df = _determine_model_accuracy(self._adata, self._X_labels, self._evaluation_outpath, self._N)
        
        if plot:
            _plot_model_accuracy(self._accuracy_df,
                                 self._N,
                                 nodes=self._nodes,
                                 layers=self._layers,
                                 seed=self._seed,
                                 outpath=self._evaluation_outpath,
                                 markersize=plot_markersize,
                                 approx_prescient=plot_approx_prescient,
                                 model_color=plot_model_color,
                                 y_labels=plot_y_labels,
                                 savename=plot_savename,
                                 width_ratios=plot_width_ratios,
                                 wspace=plot_wspace,
                                 figsize_width=figsize_width,
                                 figsize_height=figsize_height,
                                 figsize=figsize,
                                )

            
def _evaluate_model_accuracy(adata,
                             model_dir,
                             annoy_path,
                             plot_training=True,
                             plot_accuracy=True,
                             save_plots=True,
                             N=2,
                             device=0):
    
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = device
    
    evaluate = _Evaluator(model_dir)
    evaluate.load()
    if plot_training:
        evaluate.plot_training()
    
    evaluate.pass_test_data_to_model(adata, N=N, device=device)
    evaluate.calculate_cell_fate_bias(annoy_path=annoy_path)
    evaluate.determine_model_accuracy(plot=plot_accuracy, plot_savename=save_plots)
    
    return evaluate
            