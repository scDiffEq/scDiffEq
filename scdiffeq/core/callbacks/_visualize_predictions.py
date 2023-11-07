

import autodevice

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import lightning
import scdiffeq_plots as sdq_pl
from tqdm.notebook import tqdm


from .. import utils


from typing import Dict
NoneType = type(None)
import ABCParse



class VisualizePredictions(lightning.Callback, ABCParse.ABCParse):
    def __init__(
        self,
        reducer,
        kNN,
        adata,
        use_key,
        t,
        t0_idx,
        N=200,
        frequency=1,
        outdir=None,
        device=autodevice.AutoDevice(),
        label_cmap=None,
        label_key = None,
        simulation_cmap = matplotlib.cm.tab20.colors,
        *args,
        **kwargs,
    ):

        # will have to come from model.reducer.X_umap in the future (could also come from adata)
        X_umap = reducer.X_umap

        self.__parse__(locals(), private=[None])
        self._configure_time(t)
        

    def _configure_time(self, t):
        self.t = t.to(self.device)
        self.t_cmap = sdq_pl.temporal_cmap(t=t)
    
    @property
    def X0(self):
        return utils.fetch_format(
            self.adata, use_key=self.use_key, idx=self.t0_idx, N=self.N
        )

    def _FORWARD(self, pl_module, X0, t):

        X_hat = {}
        X_hat_kNN = {}
        X_hat_UMAP = {}
        for i in tqdm(range(len(self.t0_idx)), desc="Simulating", leave=False):
            key = self.t0_idx[i]
            X_hat[key] = xh = pl_module(X0[i], self.t)
            X_hat_kNN[key] = self.kNN(
                xh, annot_key=self.label_key, query_t=None
            )
            X_hat_UMAP[key] = np.stack(
                [
                    self.reducer.UMAP.transform(xh_j)
                    for xh_j in xh.detach().cpu().numpy()
                ]
            )
        return X_hat, X_hat_kNN, X_hat_UMAP

    def _background(self):

        xu = self.X_umap
        fig, axes = sdq_pl.plot(nplots=2, ncols=2, rm_ticks=True, delete_spines=["all"])

        for ax in axes:
            ax.scatter(xu[:, 0], xu[:, 1], c="lightgrey", s=2, alpha=0.5, zorder=0)

        return fig, axes

    def _label(self, X_hat_kNN, idx):
        # if you want a single label, uncomment below
        # .value_counts().idxmax()
        return X_hat_kNN[idx].idxmax(1)

    @property
    def SAVE_PATH(self):
        fname = f"./sample_model_output.epoch_{self.epoch}.png"
        if not isinstance(self.outdir, NoneType):
            return os.path.join(self.outdir, fname)
        return fname

    def _color_outline_by_simulation(self, ax, xu_x, xu_y, i, t, sizes):
        ax.scatter(
                    xu_x,
                    xu_y,
                    color=self.simulation_cmap[i],
                    zorder=t + 1,
                    s=sizes[0],
                    ec="None",
                )
        ax.scatter(
                    xu_x,
                    xu_y,
                    c="w",
                    zorder=t + 2,
                    s=sizes[1],
                    ec="None",
                )
    
    def plot(self, X_hat_umap: Dict, X_hat_kNN, sizes=[80, 40, 20]):

        fig, axes = self._background()

        for i, (idx, X_umap) in enumerate(X_hat_umap.items()):
            for t in range(len(X_umap)):
                if i == 0:
                    label = str(round(self.t[t].item(), 3))
                else:
                    label = None
                if t == 0:
                    xu_x, xu_y = X_umap[t][:, 0].mean(), X_umap[t][:, 1].mean()
                else:
                    xu_x, xu_y = X_umap[t][:, 0], X_umap[t][:, 1]

                self._color_outline_by_simulation(axes[0], xu_x, xu_y, i, t, sizes)
                self._color_outline_by_simulation(axes[1], xu_x, xu_y, i, t, sizes)
                
                axes[0].scatter(
                    xu_x,
                    xu_y,
                    color=self.t_cmap[t],
                    zorder=t + 3,
                    s=sizes[2],
                    ec="None",
                    label=label,
                )
                if not isinstance(self.label_cmap, NoneType):
                    color = X_hat_kNN[idx][t].map(self.label_cmap)
                    if t == 0:
                        color = color[0]
                else:
                    color = "dimgrey"

                axes[1].scatter(
                    xu_x, xu_y, c=color, ec="None", zorder=t + 3
                )
        if len(self.t) <= 5:
            axes[0].legend(edgecolor="None", facecolor="None")
            
            
        return fig, axes

    def __call__(self, DiffEq, t=None, sizes=[80, 40, 20], resimulate=True):

        if not isinstance(t, NoneType):
            self._configure_time(t)

        X_hat, X_hat_kNN, X_hat_umap = self._FORWARD(
            DiffEq.to(self.device), self.X0, self.t
        )
        fig, axes = self.plot(X_hat_umap, X_hat_kNN, sizes=sizes)
        plt.savefig(self.SAVE_PATH)
        plt.close()

    def on_fit_start(self, trainer, pl_module, *args, **kwargs):        
        self.epoch = pl_module.current_epoch
        
        self.__call__(pl_module)

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
        self.epoch = pl_module.current_epoch
        
        if self.epoch % self.frequency == 0:
            self.__call__(pl_module)        
