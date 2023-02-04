

def normalize_time(adata, t_min, t_max, time_key="t"):

    """Normalize time to a specified range"""
    
    adata.obs[f"_{time_key}"] = t = adata.obs[time_key]
    init_range = t.max() - t.min()
    init_norm = (t - t.min()) / init_range  # identity, if already norm'd [0,1]
    dest_range = t_max - t_min
    dest_norm = (init_norm * dest_range) + t_min
    adata.obs[time_key] = dest_norm