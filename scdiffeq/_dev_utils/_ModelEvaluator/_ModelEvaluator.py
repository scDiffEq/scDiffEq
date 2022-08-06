

import os
import torch


from ._funcs import _get_best_epoch
from ._funcs import _training_summary
from ._funcs import _plot_training_metrics
from ._funcs import _load_ckpt_state

from ..._preprocessing import lazy_LARRY

class ModelEvaluator:
        
    def __init__(self,
                 model_name,
                 seed=0,
                 h5ad_path="/home/mvinyard/data/Weinreb2020.adata.h5ad",
                 model_outs_basename="/home/mvinyard/notebooks/benchmark/outs/",
                 base_src_path = "/home/mvinyard/notebooks/benchmark/src/",
                 task="timepoint_recovery",
                 device="cuda:0",
                ):

        if torch.cuda.is_available():
            self._device = device
        else:
            self._device = "cpu"
                
        self._seed = seed
        self._task = task
        self._h5ad_path = h5ad_path
        self._base_src_path = base_src_path
        self._src_script_path = os.path.join(self._base_src_path, "{}.src.py".format(model_name))
        self._model_path = os.path.join(model_outs_basename, model_name)
        self._summary = _training_summary(self._model_path, self._seed)
        self._best_epoch, self._best_epoch_path = _get_best_epoch(self._summary)
        try:
            self._best_score = float(summary_dict["best_model_score"])
        except:
            pass

    def plot(self, smooth=5):

        _plot_training_metrics(self._model_path, seed=self._seed, smooth=smooth)
        
    def load_best_ckpt(self):
                
        self._adata, self._dataset = lazy_LARRY(self._h5ad_path, self._task)
        self.best_model = _load_ckpt_state(self._adata,
                         self._src_script_path,
                         self._best_epoch_path,
                         self._device,
                        )
        self.best_model._dataset = self._dataset
        
        