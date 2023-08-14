
import torch
from autodevice import AutoDevice
import adata_query


# -- Controller class: -----
class StateCharacterization:
    """Quantify the instantaneous drift / diffusion given a state and model."""
    def __init__(self, SDE, scalar=1):

        self.SDE = SDE
        self.scalar = scalar
    
    @property
    def t(self):
        return None
    
    def drift(self, X: torch.Tensor):
        return self.SDE.f(self.t, X).squeeze(dim=-1)
    
    def diffusion(self, X: torch.Tensor):
        return self.SDE.g(self.t, X).squeeze(dim=-1)
    
    
# -- API-facing functions: -----
def drift(adata, SDE, use_key="X_pca", key_added="X_drift", return_Tensor: bool = False):
    """Accessed as sdq.tl.drift(adata)"""
    
    X = adata_query.fetch(adata, key=use_key)
    SDE_state = StateCharacterization(SDE)
    X_drift = SDE_state.drift(X)
    adata.obsm[key_added] = X_drift.detach().cpu().numpy()
    
    if not (key_added is None):
        adata.obsm[key_added] = X_drift.detach().cpu().numpy()
    
    if return_Tensor:
        return X_drift
    
    
def diffusion(adata, SDE, use_key="X_pca", key_added="X_diffusion", return_Tensor: bool = False):
    """Accessed as sdq.tl.diffusion(adata)"""
    
    X = fetch(adata, use_key=use_key)
    SDE_state = StateCharacterization(SDE)
    X_diffusion = SDE_state.diffusion(X)
    
    if not (key_added is None):
        adata.obsm[key_added] = X_diffusion.detach().cpu().numpy()
    
    if return_Tensor:
        return X_diffusion
