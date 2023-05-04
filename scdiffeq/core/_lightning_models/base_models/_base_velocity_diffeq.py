
# -- import packages: -------------------------------------------------
import torch
from autodevice import AutoDevice


# -- import local dependencies: --------------------------------------
from ._base_lightning_diffeq import BaseLightningDiffEq


# -- module class: ----------------------------------------------------
class BaseVeloDiffEq(BaseLightningDiffEq):
    from sklearn.decomposition import PCA
    def __init__(self, pca=None, velo_gene_idx=None, gene_space_key="X_scaled", dt=0.1, n_pcs=50):
        super(BaseVeloDiffEq, self).__init__()
                
    @property
    def pca_model(self):
        if isinstance(self._pca_model, NoneType):            
            
            print("doing PCA...", end="")
            self._pca_model = self.PCA(n_components=self.hparams['n_pcs'])
            X_gene = self.adata.layers[self.hparams['gene_space_key']]
            self._pca_model.fit_transform(X_gene)
            print("done")
            
        return self._pca_model
        
    def inferred_velocity(self, X_hat):
        
        """
        Inferred instantaneous velocity at a given position from the delta 
        of the observed positions
        """

        X_hat_np = X_hat.detach().cpu().numpy()
        X_hat_ = torch.stack(
            [torch.Tensor(self.pca_model.inverse_transform(xt)) for xt in X_hat_np]
        ).to(AutoDevice())
        X_hat_ = X_hat_[:, :, self.hparams['velo_gene_idx']]
        
        return torch.diff(X_hat_, n=1, dim=0, append=X_hat_[-1][None, :, :])
    
    def loss(self, W, X, W_hat, X_hat, V, V_hat):
        
        state_loss = self.loss_func(W, X, W_hat, X_hat)
        X = torch.concat([X, V], axis=-1).contiguous()
        X_hat = torch.concat([X_hat, V_hat], axis=-1).contiguous()
        velo_loss = self.loss_func(W, X, W_hat, X_hat)

        return {"state": state_loss, "velo": velo_loss}
