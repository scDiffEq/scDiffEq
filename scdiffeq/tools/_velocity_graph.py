
import ABCParse
import anndata

from .utils import ComputeCosines, scverse_neighbors


from typing import Dict, Optional


# -- API-facing function: ------------------------------------------------
def velocity_graph(
    adata: anndata.AnnData,
    state_key: str = "X_pca",
    velocity_key: str = "X_drift",
    n_pcs: Optional[int] = None,
    velocity_key_added: str = "velocity",
    split_negative: bool = True,
    silent: bool = False,
    neighbor_kwargs: Dict = {},
    *args,
    **kwargs,
):
    """
    Args:
        adata (anndata.AnnData)
        
        state_key (str): ... **Default** = "velocity"
        
        velocity_key (str): ... **Default** = "X_drift"
        
        velocity_key_added (str): ... **Default** = "velocity"
        
        split_negative (bool): ... **Default** = True
        
        silent (bool): **Default** = False.
        
        neighbor_kwargs (Optional[Dict[str, Any]]): **Default** = {},
        
    
    Returns:
        (None)
    """
    
    scverse_neighbors(adata, silent=silent, **neighbor_kwargs)
    
    init_kw = ABCParse.function_kwargs(ComputeCosines.__init__, locals())
    call_kw = ABCParse.function_kwargs(ComputeCosines.__call__, locals())

    compute_cosines = ComputeCosines(**init_kw)
    compute_cosines(**call_kw)
