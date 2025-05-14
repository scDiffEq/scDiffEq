# -- import packages: ---------------------------------------------------------
import ABCParse
import anndata
import time
import logging

 # -- import local dependencies: ----------------------------------------------
from . import utils

# -- set type hints: ----------------------------------------------------------
from typing import Dict, Optional

# -- configure logger: --------------------------------------------------------
logger = logging.getLogger(__name__)

# -- API-facing function: -----------------------------------------------------
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
    
    start_time = time.time()
    logger.debug("Starting velocity_graph computation...")
    
    utils.scverse_neighbors(adata, silent=silent, **neighbor_kwargs)
    
    init_kw = ABCParse.function_kwargs(utils.ComputeCosines.__init__, locals())
    call_kw = ABCParse.function_kwargs(utils.ComputeCosines.__call__, locals())

    logger.debug("Starting ComputeCosines computation...")
    cc_start = time.time()
    compute_cosines = utils.ComputeCosines(**init_kw)
    compute_cosines(**call_kw)
    cc_end = time.time()
    logger.debug(f"ComputeCosines computation finished in {cc_end - cc_start:.2f} seconds.")
    logger.debug(f"velocity_graph finished in {time.time() - start_time:.2f} seconds.")
