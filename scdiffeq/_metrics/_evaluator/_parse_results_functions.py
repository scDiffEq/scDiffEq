
import numpy as np

def _return_epochs_to_evaluate(log_df, n_epochs=5):
    
    saved_df = log_df.loc[log_df['saved']==True]
    saved_df = saved_df.reset_index(drop=True)
    idx = np.linspace(0, saved_df.shape[0]-1, n_epochs, dtype=int)
    epochs_to_evaluate = saved_df.iloc[idx]['epoch'].values
    
    return epochs_to_evaluate