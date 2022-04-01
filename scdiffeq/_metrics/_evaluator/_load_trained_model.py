import glob
import numpy as np
import os
import pandas as pd

def _parse_run_signature(self):
    
    """"""
    
    run_signature = os.path.basename(os.path.dirname(self._path))    
    
    self._seed = run_signature.split(":")[-1]
    self._nodes = int(run_signature.split("nodes")[0].split("_")[-1])
    self._layers = int(run_signature.split("layers")[0].split("_")[-1])
    self._run_signature = run_signature
    
def _parse_model_training_path(self, path):
    
    self._path = path
    _parse_run_signature(self)
    self._log_path = self._path + "status.log"
    self._model_training_dir = os.path.join(self._path, "model/*")
    self._training_model_paths = model_paths = glob.glob(self._model_training_dir)
    
def _get_saved_model_epochs(model_paths):
        
    saved_model_epochs = []
    
    for path in model_paths:
        epoch = int(os.path.basename(path).split("_")[0])
        saved_model_epochs.append(epoch)
        
    return saved_model_epochs

def _mark_saved_epochs(log_df, saved_model_epochs):
    
    saved_vector = np.full(len(log_df), False)
    saved_vector[saved_model_epochs] = True
    log_df['saved'] = saved_vector
    
    return log_df

def _load_model_log_df(path):
    return pd.read_csv(path, sep='\t')
    
def _get_best_saved_epoch(log_df):
    
    best_saved_epoch = log_df.loc[log_df['saved'] == True]['total'].idxmin()
    best_vector = np.full(len(log_df), False)
    best_vector[best_saved_epoch] = True
    log_df['best'] = best_vector
    
    return log_df

def _load_trained_model(log_path, training_model_paths):
    
    log_df = _load_model_log_df(log_path)
    saved_model_epochs = _get_saved_model_epochs(training_model_paths)
    log_df = _mark_saved_epochs(log_df, saved_model_epochs)
    log_df = _get_best_saved_epoch(log_df)
    
    return log_df