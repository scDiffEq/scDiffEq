import anndata as a
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_array(adata):

    """
    Parameters:
    -----------
    adata
        AnnData object

    Returns:
    --------
    None, modified in place.
    """

    try:
        adata.X = adata.X.toarray()
    except:
        pass
    

def _load_development_libraries():
    
    """
    Assigns global variables to packages used in the development of nodescape such that developing with common packages can be done cleanly.
    
    Parameters:
    -----------
    None
    
    Returns:
    --------
    odeint, torch, np, pd, plt, nn, a, os, time, optim, sp, PCA
        type(s): modules
        
    Notes:
    ------
    (1) To implement this function, copy-paste the following:
        
        odeint, torch, np, pd, plt, nn, a, os, time, optim, sp, PCA, v = sdq.ut.devlibs()
        
    """

    global torch
    global time
    global np
    global pd
    global plt
    global nn
    global a
    global os
    global optim
    global odeint
    global sp
    global PCA
    global v
    
    import torch as torch
    from torchdiffeq import odeint
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch.nn as nn
    import torch.optim as optim
    import anndata as a
    import os
    import time
    import scipy as sp
    from sklearn.decomposition import PCA
    import vintools as v

    
    return odeint, torch, np, pd, plt, nn, a, os, time, optim, sp, PCA, v



# from .._tools._pca import pca

def _use_embedding(adata, emb="X_pca"):

    """use the dimensionally reduced matrix as the adata.X data matrix."""

    try:
        adata.obsm["X_pca"]
    except:
        pca(adata)

    bdata = a.AnnData(adata.obsm[emb])
    bdata.obs = adata.obs
    bdata.uns = adata.uns
    pca(bdata)
    
    return bdata