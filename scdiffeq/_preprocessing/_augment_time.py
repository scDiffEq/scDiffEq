
import pandas as pd

def _augment_time(adata,
                  TimeConvDict={2: 0, 4: 0.01, 6: 0.02},
                  time_key="Time Key",
                  key_added="t",
                 ):

    """
    
    annotate adata.obs with an augmented time, t
    
    """

    time_df = pd.DataFrame.from_dict(TimeConvDict, orient="index").reset_index()
    time_df.columns = [time_key, key_added]
    adata.obs = adata.obs.merge(time_df, on=time_key)