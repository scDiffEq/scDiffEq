# -- import packages: --------------------------------------------------------------------
from lightning import Callback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import vinplots
import os


# -- Helper functions: -------------------------------------------------------------------
def _epoch_regex_mean(epoch_df, regex):
    return epoch_df.filter(regex=regex).dropna().mean(0)


# -- Plotting: ---------------------------------------------------------------------------
class LossPlotter(vinplots.Plot):
    def __init__(self, time_unit="", x_label="Epoch", y_label="Wasserstein Distance"):

        self.build()
        self.time_unit = time_unit
        self._colors = np.array(list(vinplots.colors.pbmc3k.values()))
        # self.colors = self._colors[[4, 6, 7]]
        cc = vinplots.colors.BuOr.colors
        self.colors = cc[:int(len(cc)/2)][::3]
        self.x_label = x_label
        self.y_label = y_label
        self.y_min = 0
        self.y_max = 300
        self.legend_fontsize = "small"
        self.fit_colors = {"train": self._colors[0], "val": self._colors[2]}

    def build(self):

        self.construct(nplots=3, ncols=3, wspace=0.1, figsize=0.8)
        self.modify_spines(ax="all", spines_to_delete=["top", "right"])
        self.axes = self.linearize()

    def _format_ax(self, ax, title):

        ax.tick_params(axis="both", which="both", labelsize=6)
        ax.set_xlabel(self.x_label, fontsize=8)
        ax.set_ylabel(self.y_label, fontsize=8)
#         ax.set_ylim(self.y_min, self.y_max)
        ax.set_title(title, fontsize=10)
        ax.grid(True, zorder=0, alpha=0.2)

    def _plot_ax(self, ax, loss_df, fit=False, title=None):

        if not fit:
            iterable = loss_df.columns[1:-1]
        else:
            iterable = loss_df.columns
        
        for n, i in enumerate(iterable):
            if isinstance(i, int):
                label = "d{}".format(i)
            else:
                label = i
            ax.plot(loss_df[i], c=self.colors[n], label=label)
            if n == (len(iterable)-1):
                ax.legend(edgecolor="w", fontsize=self.legend_fontsize)
        self._format_ax(ax, title)

        
    def forward(self, loss, savepath):

        self._plot_ax(self.axes[0], loss.train, fit=False, title="Training Loss")
        self._plot_ax(self.axes[1], loss.val, fit=False, title="Validation Loss")
        self._plot_ax(self.axes[2], loss.fit, fit=True,  title="Fitting Loss")

        if savepath:
            plt.savefig(savepath + ".svg")
            plt.savefig(savepath + ".png")


# -- Get loss: ---------------------------------------------------------------------------
class LossReader:
    def __init__(self, trainer):

        self._log_dir = trainer.logger.log_dir
        self._metrics_path = os.path.join(self._log_dir, "metrics.csv")
    
    @property
    def metrics_csv_exists(self):
        return os.path.exists(self._metrics_path)
    
    @property
    def metrics_df(self):
        if self.metrics_csv_exists:
            return pd.read_csv(self._metrics_path)
    
    def _adjust_colnames(self, filtered_loss_df):

        cols = [int(col.split("_")[1]) for col in filtered_loss_df.columns]
        filtered_loss_df.columns = cols
        filtered_loss_df["sum"] = filtered_loss_df.sum(1).values

        return filtered_loss_df

    def filter_loss_df(self, regex, groupby="epoch"):

        filtered_loss_df = (
            self.metrics_df.groupby(groupby).apply(_epoch_regex_mean, regex=regex).copy()
        )
        return self._adjust_colnames(filtered_loss_df).dropna()

    @property
    def train(self) -> pd.DataFrame:
        return self.filter_loss_df(regex="train")

    @property
    def val(self) -> pd.DataFrame:
        return self.filter_loss_df(regex="val")

    @property
    def fit(self) -> pd.DataFrame:
        return pd.DataFrame({"train": self.train["sum"], "val": self.val["sum"]})

    def plot(self, savepath=None):

        if self.metrics_csv_exists:
            self.plotter = LossPlotter()
            self.plotter.forward(self, savepath=savepath)


# -- Callback: ---------------------------------------------------------------------------
class LossAccounting(Callback):
    def __init__(self, save_img=True):
        
        self.save_img = save_img
        self.savepath = None
    
    def _forward(self, trainer):
        
        loss = LossReader(trainer)
        if loss.metrics_csv_exists:
            if self.save_img:
                self.savepath = os.path.join(trainer.logger.log_dir, "model_fit_loss")
            loss.plot(savepath=self.savepath)
        
    def on_validation_epoch_end(self, trainer, model):
        
        self._forward(trainer)
        plt.close();
        
    def on_fit_end(self, trainer, model):
        
        self._forward(trainer)
