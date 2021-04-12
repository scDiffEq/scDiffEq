
def ensure_array(adata):

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
    

def load_development_libraries():
    
    """
    Assigns global variables to packages used in the development of nodescape such that developing with common packages can be done cleanly.
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

    
    return odeint, torch, np, pd, plt, nn, a, os, time, optim, sp, PCA, 