
__module_name__ = "_interpolation_test.py"
__doc__ = """Callback for passing the test set to measure interpolation of a withheld timepoint."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])


# -- import packages: ----------------------------------------------------------
from pytorch_lightning import Callback
import numpy as np
import torch


# -- import local dependencies: ------------------------------------------------
from ..lightning_models import SinkhornDivergence


# -- Main callback class: ------------------------------------------------------
class InterpolationTest(Callback):
    def __init__(self, adata, potential=False, seed: int = 617):
        
        if isinstance(seed, int):
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.adata = adata
        self.Loss = SinkhornDivergence()
        self.configure_eval_data()
        self.potential = potential

    def configure_eval_data(self):

        self.df = self.adata.obs.copy()
        self.df_clonal = self.df.loc[self.df["clone_idx"].notna()]
        self.t0_idx = (
            self.df_clonal.loc[self.df_clonal["Time point"] == 2]
            .sample(10_000, replace=True)
            .index
        )
        self.X0 = torch.Tensor(self.adata[self.t0_idx].obsm["X_pca"]).to("cuda:0")
        self.t = torch.Tensor([2, 4, 6]).to("cuda:0")
        
        # test
        self.X_d4 = torch.Tensor(
            self.adata[self.df_clonal.loc[self.df_clonal["Time point"] == 4].index].obsm["X_pca"]
        ).to("cuda:0")
        
        # train
        self.X_d6 = torch.Tensor(
            self.adata[self.df_clonal.loc[self.df_clonal["Time point"] == 6].index].obsm["X_pca"]
        ).to("cuda:0")
        
    def forward_without_grad(self, DiffEq):
        """Forward integrate over the model without gradients."""
        with torch.no_grad():
            return DiffEq.forward(self.X0, self.t)
            
    def forward_with_grad(self, DiffEq):
        """Forward integrate over the model retaining gradients."""
        torch.set_grad_enabled(True)
        return DiffEq.forward(self.X0, self.t)
        

    def on_train_epoch_end(self, trainer, DiffEq):
        """Call"""
        if self.potential:
            X_hat = self.forward_with_grad(DiffEq)
        else:
            X_hat = self.forward_without_grad(DiffEq)
        
        d4_loss = self.Loss(X_hat[1], self.X_d4)
        d6_loss = self.Loss(X_hat[2], self.X_d6)
        
        self.log(f"eval_d4_loss", d4_loss.item())
        self.log(f"eval_d6_loss", d6_loss.item())
