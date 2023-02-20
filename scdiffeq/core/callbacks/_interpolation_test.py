

from pytorch_lightning import Callback
import numpy as np
import torch


from ..lightning_models import SinkhornDivergence


class InterpolationTest(Callback):
    def __init__(self, adata):

        torch.manual_seed(617)
        np.random.seed(617)

        self.adata = adata
        self.Loss = SinkhornDivergence()
        self.configure_test_data()

    def configure_test_data(self):

        self.df = self.adata.obs.copy()
        self.df_clonal = self.df.loc[self.df["clone_idx"].notna()]
        self.t0_idx = (
            self.df_clonal.loc[self.df_clonal["Time point"] == 2]
            .sample(10_000, replace=True)
            .index
        )
        self.X0 = torch.Tensor(self.adata[self.t0_idx].obsm["X_pca"]).to("cuda:0")
        self.t = torch.Tensor([2, 4, 6]).to("cuda:0")
        self.X_d4 = torch.Tensor(
            self.adata[self.df_clonal.loc[self.df_clonal["Time point"] == 4].index].obsm["X_pca"]
        ).to("cuda:0")

    def on_train_epoch_end(self, trainer, DiffEq):
        
        with torch.no_grad():
            X_hat = DiffEq.forward(self.X0, self.t)        
        loss = self.Loss(X_hat[1], self.X_d4)
        self.log(f"test_loss", loss.item())
