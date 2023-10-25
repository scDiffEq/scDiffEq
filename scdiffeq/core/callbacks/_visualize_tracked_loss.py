
"""
Three main classes that are going to be used here:

1. [ called in callback ] ModelTracker
2. [ called in callback ] LossTrackingVisualization 
3. [ lightning.Callback ] VisualizeTrackedLoss
"""
import lightning
import glob
import os

from .. import utils

from typing import List
import scdiffeq_plots as sdq_pl
import matplotlib.pyplot as plt
import ABCParse


class ModelTracker(ABCParse.ABCParse):
    def __init__(
        self,
        version,
        model_name="scDiffEq_model",
        working_dir=os.getcwd(),
        train_version=0,
        pretrain_version=0,
    ):
        super().__init__()
        self.__parse__(locals(), public=[None])

    @property
    def _PATH(self):
        return os.path.join(
            self._working_dir, self._model_name, f"version_{self._version}"
        )

    def _HAS_STAGE_LOGS(self, stage_key):
        for fpath in glob.glob(os.path.join(self._PATH, "*/version_*/metrics.csv")):
            if stage_key in fpath:
                return True
            

    @property
    def _HAS_PRETRAIN(self):
        return self._HAS_STAGE_LOGS("/pretrain_logs")

    @property
    def _HAS_TRAIN(self):
        return self._HAS_STAGE_LOGS("/train_logs")

    @property
    def _PRETRAIN(self):
        if self._HAS_PRETRAIN:
            return utils.PretrainLogs(
                self._PATH, stage="pretrain", version=self._pretrain_version
            )

    @property
    def _TRAIN(self):
        if self._HAS_TRAIN:
            return utils.TrainLogs(
                self._PATH, stage="train", version=self._train_version
            )

    def _subset_df_for_plotting(
        self,
        df,
        re_patterns=(
            ["pretrain_rl_mse"],
            ["sinkhorn", "fate_weighted"],
            ["sinkhorn_fate_weighted"],
            ["total"],
            ["fate_accuracy"],
            ["kl_div"],
        ),
        include_regex=[[True], [True, False], [True], [True], [True], [True]],
    ):
        """
        Returns dictionary of dfs subset for values included on individual plots.
        Filters PlotFrames Dict to return only measured metrics
        """

        PlotFrames = {}
        for regex, include in zip(re_patterns, include_regex):
            filt_df = utils.filter_df(df, regex=regex, include=include)
            measured = filt_df.shape[1] > 0
            if measured:
                PlotFrames[regex[0]] = filt_df

        return PlotFrames

    @property
    def pretrain_df(self):
        if self._HAS_PRETRAIN:
            return self._PRETRAIN()

    @property
    def train_df(self):
        if self._HAS_TRAIN:
            return self._TRAIN()

    @property
    def pretrain_plot_inputs(self):
        if self._HAS_PRETRAIN:
            return self._subset_df_for_plotting(self.pretrain_df)

    @property
    def train_plot_inputs(self):
        if self._HAS_TRAIN:
            return self._subset_df_for_plotting(self.train_df)


class LossTrackingVisualization(ABCParse.ABCParse):
    def __init__(
        self,
        tracker,
        ncols: int = 3,
        scale: float = 1,
        width: float = 1,
        height: float = 1,
        hspace: float = 0.4,
        wspace: float = 0.4,
        width_ratios: List[float] = None,
        height_ratios: List[float] = None,
        linearize=True,
        rm_ticks=False,
        color=[None],
        move=[0],
        xy_spines: bool = True,
        delete_spines=[[]],
        color_spines=[[]],
        move_spines=[[]],
        xlabel="Epoch",
        title_fontsize=10,
        label_fontsize=8,
        tick_param_size=6,
        colors=["dodgerblue", "darkorange"],
        grid=True,
        grid_kwargs={"zorder": 0, "alpha": 0.5},
        legend_kwargs={"edgecolor": "None", "facecolor": "None", "fontsize": 8},
    ):
        super().__init__()

        self.__parse__(locals(), public=[None])
        self._configure_plot_inputs()

    def _configure_plot_inputs(self):
        self._PLOT_INPUTS = {}
        if self._HAS_PRETRAIN:
            self._PLOT_INPUTS["pretrain"] = self._tracker.pretrain_plot_inputs
        self._PLOT_INPUTS["train"] = self._tracker.train_plot_inputs

    @property
    def _PLOT_TITLES(self):
        return {
            "pretrain_rl_mse": "Pretrain Reconstruction Loss (MSE)",
            "sinkhorn": "Sinkhorn Divergence",
            "sinkhorn_fate_weighted": "Fate-Weighted Sinkhorn Divergence",
            "fate_accuracy": "Fate Accuracy",
            "kl_div": "KL Divergence",
            "total": "Total Loss",
        }

    @property
    def _PLOT_LABELS(self):
        return {
            "pretrain_rl_mse": "MSE",
            "sinkhorn": "Wasserstein Distance",
            "sinkhorn_fate_weighted": "Wasserstein Distance",
            "fate_accuracy": "Accuracy Score",
            "kl_div": "KL Divergence",
            "total": "Mixed Units",
        }

    @property
    def _PLOT_KWARGS(self):
        return utils.extract_func_kwargs(func=sdq_pl.plot, kwargs=self._PARAMS)

    @property
    def _HAS_TRAIN(self):
        return self._tracker._HAS_TRAIN
    
    @property
    def _HAS_PRETRAIN(self):
        return self._tracker._HAS_PRETRAIN

    @property
    def _PLOT_KEYS(self):
        
        keys = []
        if self._HAS_PRETRAIN:
            keys+= list(self._PLOT_INPUTS["pretrain"].keys())
            
        if self._HAS_TRAIN:
            keys+= self._PLOT_INPUTS["train"].keys()
            
        return keys

    @property
    def _NPLOTS(self):
        nplots = 0        
        if self._HAS_TRAIN:
            nplots += len(self._PLOT_INPUTS["train"].keys())
        if self._HAS_PRETRAIN:
            nplots += len(self._PLOT_INPUTS["pretrain"].keys())
        print(f"nplots: {nplots}")
        return nplots

    def __layout__(self):
        
        self.fig, self.axes = sdq_pl.plot(
            nplots=self._NPLOTS,
            **self._PLOT_KWARGS,
        )
        for en, ax in enumerate(self.axes):
            ax.set_title(
                self._PLOT_TITLES[self._PLOT_KEYS[en]], fontsize=self._title_fontsize
            )
            ax.set_ylabel(
                self._PLOT_LABELS[self._PLOT_KEYS[en]], fontsize=self._label_fontsize
            )
            ax.set_xlabel(self._xlabel, fontsize=self._label_fontsize)
            ax.tick_params(axis="both", which="both", labelsize=self._tick_param_size)

    def __plot__(self):

        self.__layout__()

        self._COMBINED_PLOT_INPUTS = {}

        if self._HAS_PRETRAIN:
            self._COMBINED_PLOT_INPUTS = self._PLOT_INPUTS["pretrain"]
        if self._HAS_TRAIN:
            self._COMBINED_PLOT_INPUTS.update(self._PLOT_INPUTS["train"])

        for en, ax in enumerate(self.axes):
            val = self._COMBINED_PLOT_INPUTS[self._PLOT_KEYS[en]]
            for col_i, col in enumerate(val.columns):
                self.axes[en].plot(val[col], color=self._colors[col_i], label=col, lw=2)
                if self._grid:
                    self.axes[en].grid(**self._grid_kwargs)
            self.axes[en].legend(**self._legend_kwargs)


class VisualizeTrackedLoss(lightning.Callback):
    def __init__(
        self,
        version,
        viz_frequency = 1,
        model_name="scDiffEq_model",
        working_dir=os.getcwd(),
        train_version=0,
        pretrain_version=0,
        fname = "scDiffEq_fit_loss_tracking.png",
        *args,
        **kwargs,
    ):
        self.model_tracker = ModelTracker(
            **utils.extract_func_kwargs(func=ModelTracker, kwargs=locals())
        )
        self._INFO = utils.InfoMessage()
        self.fname = fname
        self.viz_frequency = viz_frequency
        
    @property
    def _VIZ_DISABLE(self):
        return not self.vis_frequency is None
        
    @property
    def save_path(self):
        return os.path.join(self.model_tracker._PATH, self.fname)
        
    def _save_plot(self):
        if not os.path.exists(self.save_path):
            self._INFO(f"Loss visualization saved to: {self.save_path}")
        plt.savefig(self.save_path)
        plt.close()
        
    def on_train_epoch_end(self, trianer, pl_module, *args, **kwargs):
        
        epoch = pl_module.current_epoch
        
        if not self._VIZ_DISABLED and (epoch % self.viz_frequency == 0):
            
            loss_track_viz = LossTrackingVisualization(self.model_tracker)
            loss_track_viz.__plot__()
            self._save_plot()
