
import numpy as np


from ._annotate_cells import annotate_cells


class TimeFreeSampling:
    def __init__(self, adata, n_steps, t0_key="t0"):
        self.__parse__(locals())
        
    def __parse__(self, kwargs, ignore=['self']):
        
        for key, val in kwargs.items():
            if not key in ignore:
                setattr(self, key, val)        
        
    def annotate_t0(self, t0_idx):
        annotate_cells(self.adata, idx=t0_idx, key=self.t0_key)
        
    @property
    def non_t0_idx(self):
        df = self.adata.obs.copy()
        return df.loc[~df[self.t0_key]].index
    
    @property
    def stepwise_samples(self):
        return np.random.choice(range(1, self.n_steps), self.non_t0_idx.shape)
    
    @property
    def time_free_sampled_groups(self):
        idx = np.zeros(len(self.adata))
        # organize remaining non-t0 cells into n_steps # of groups
        idx[self.non_t0_idx.astype(int)] = self.stepwise_samples
        return idx
    
def time_free_sampling(adata, t0_idx, n_steps, t0_key="t0", t_key="t"):
    """
    Notes:
    ------
    Could potentially improve this through a probabalistic sampling
    processe, weighting certain cells to be more likely assigned to
    'later' groups.
    """
    tfs = TimeFreeSampling(adata, n_steps=n_steps, t0_key=t0_key)
    tfs.annotate_t0(t0_idx)
    adata.obs[t_key] = tfs.time_free_sampled_groups
    adata.obs[t_key] = (adata.obs[t_key] / adata.obs[t_key].max())
