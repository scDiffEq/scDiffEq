import torch
import numpy as np
import matplotlib.pyplot as plt
import vintools as v
import os
from IPython.display import display

def _get_n_by_training_group(adata, trajectory_key="trajectory"):

    GroupSizes = {}

    for group in ["train", "valid", "test"]:
        GroupSizes[group] = adata.obs.loc[adata.obs[group] == True][
            trajectory_key
        ].nunique()

    return GroupSizes


def _get_y0_idx(df, time_key):

    """"""

    y0_idx = df.index[np.where(df[time_key] == df[time_key].min())]

    return y0_idx


def _get_adata_y0(adata, time_key):

    y0_idx = _get_y0_idx(adata.obs, time_key)
    adata_y0 = adata[y0_idx].copy()

    return adata_y0


def _get_y0(adata, use, time_key):

    adata_y0 = _get_adata_y0(adata, time_key)

    if use == "X":
        return torch.Tensor(adata_y0.X)

    elif use in adata.obsm_keys():
        return torch.Tensor(adata_y0.obsm[use])

    else:
        print("y0 not properly defined!")


def _fetch_data(adata, use="X", time_key="time"):

    """

    Assumes parallel time.
    """

    y = torch.Tensor(adata.X)
    y0 = _get_y0(adata, use, time_key)
    t = torch.Tensor(adata.obs[time_key].unique())

    return y, y0, t


def _format_valid_predictions_for_plotting(evaluator, group):

    x, y = evaluator.y[group].detach().numpy()[:, 0].T, evaluator.y[group].detach().numpy()[:, 1].T
    x_, y_ = (
        evaluator.pred_y[group].detach().numpy()[:, :, 0].T,
        evaluator.pred_y[group].detach().numpy()[:, :, 1].T,
    )

    return [x, y, x_, y_]

def _tile_time(time, pred):
    return np.tile(time, pred.shape[1])

def _size_adjust_loss_values(loss_vector, GroupSizesDict, group):
    return np.array(loss_vector)*100 / GroupSizesDict[group]

def _scale_loss_values_for_ploting(TrainingMonitor, validator):
    
    size_adjusted_train_loss = _size_adjust_loss_values(
        TrainingMonitor.train_loss, validator.GroupSizesDict, group="train"
    )
    size_adjusted_valid_loss = _size_adjust_loss_values(
        TrainingMonitor.valid_loss, validator.GroupSizesDict, group="valid"
    )
        
    scaled_train_loss = size_adjusted_train_loss / size_adjusted_train_loss.max()
    scaled_valid_loss = size_adjusted_valid_loss / size_adjusted_valid_loss.max()
    
    return scaled_train_loss, scaled_valid_loss

def _prepare_binned_loss(loss_vector, binsize):

    """"""

    BinnedMeanStd = {}
    BinnedMeanStd["mean"] = []
    BinnedMeanStd["std"] = []
    
    if len(loss_vector) < binsize:
        bin_i = 0
        return BinnedMeanStd, bin_i
    else:
        for bin_i in range(int(len(loss_vector) / binsize)):
            bin_start = bin_i * binsize
            bin_stop = bin_i * binsize + binsize
            binned = np.array(loss_vector)[bin_start:bin_stop]
            BinnedMeanStd["mean"].append(binned.mean())
            BinnedMeanStd["std"].append(binned.std())

    return BinnedMeanStd, bin_i

def _plot_loss(ax, train_loss, binsize, valid_loss, validation_frequency):
    
    BinnedMeanStd, max_bin = _prepare_binned_loss(train_loss, binsize)
    x_range = np.arange(len(BinnedMeanStd["mean"]))*binsize

    y_mean = np.array(BinnedMeanStd["mean"])
    y1 = y_mean + np.array(BinnedMeanStd["std"])
    y2 = y_mean - np.array(BinnedMeanStd["std"])

    ax.fill_between(x_range, y1, y2, color="navy", alpha=0.2, zorder=1)
    ax.plot(x_range, y_mean, c="navy", zorder=2, label="Training")
    
    x_range_valid = np.arange(len(valid_loss))*validation_frequency
    ax.plot(x_range_valid, valid_loss, c="orange", zorder=2, label="Validation")
    
import time
def _plot_validation(validator, TrainingMonitor, HyperParameters, binsize, title_fontsize, save_path):
    
#     plot_time_start = time.time()
    
    plot = v.pl.ScatterPlot()
    plot.construct_layout(nplots=8, ncols=4, grid_hspace=0.4, width_ratios=np.ones(4), figsize_height=1.4, figsize_width=1)
    plot.style()

    ax_train = plot.AxesDict[0][0]
    ax_train_predicted = plot.AxesDict[0][1]
    ax_train_overlaid = plot.AxesDict[0][2]
    ax_loss_curve = plot.AxesDict[0][3]
    
    ax_valid = plot.AxesDict[1][0]
    ax_valid_predicted = plot.AxesDict[1][1]
    ax_valid_overlaid = plot.AxesDict[1][2]
    
    plot.AxesDict[1][3].remove()
    
    [xt, yt, xt_, yt_] = _format_valid_predictions_for_plotting(validator, group="train")
    [x, y, x_, y_] = _format_valid_predictions_for_plotting(validator, group="valid")
    t_valid = _tile_time(validator.t, validator.pred_y['valid'])
    t_train = _tile_time(validator.t, validator.pred_y['train'])
    
    ax_train.scatter(xt, yt, c=t_train, zorder=1)
    ax_train_predicted.scatter(xt_, yt_, c=t_train, zorder=1)
    ax_train_overlaid.scatter(xt, yt, c="lightgrey", alpha=0.75, zorder=1)
    ax_train_overlaid.scatter(xt_, yt_, c=t_train, zorder=2)
    
    ax_train.set_title("Training Data", fontsize=title_fontsize, y=1.06)
    ax_train_predicted.set_title("Training Predictions", fontsize=title_fontsize, y=1.06)
    ax_train_overlaid.set_title("Training Predictions Overlaid", fontsize=title_fontsize, y=1.06)
    
    ax_valid.scatter(x, y, c=t_valid, zorder=1)
    ax_valid_predicted.scatter(x_, y_, c=t_valid, zorder=1)
    ax_valid_overlaid.scatter(x, y, c="lightgrey", alpha=0.75, zorder=1)
    ax_valid_overlaid.scatter(x_, y_, c=t_valid, zorder=2)
    
    ax_loss_curve.set_title("Training and Validation Loss", fontsize=title_fontsize, y=1.06)
    
    ax_valid.set_title("Validation Data", fontsize=title_fontsize, y=1.06)
    ax_valid_predicted.set_title("Validation Predictions", fontsize=title_fontsize, y=1.06)
    ax_valid_overlaid.set_title("Validation Predictions Overlaid", fontsize=title_fontsize, y=1.06)
    
    ax_train.set_xlabel("$x$")
    ax_train.set_ylabel("$y$")
    ax_train_predicted.set_xlabel("$x$")
    ax_train_predicted.set_ylabel("$y$")
    ax_train_overlaid.set_xlabel("$x$")
    ax_train_overlaid.set_ylabel("$y$")
    
    ax_valid.set_xlabel("$x$")
    ax_valid.set_ylabel("$y$")
    ax_valid_predicted.set_xlabel("$x$")
    ax_valid_predicted.set_ylabel("$y$")
    ax_valid_overlaid.set_xlabel("$x$")
    ax_valid_overlaid.set_ylabel("$y$")
    
    scaled_train_loss, scaled_valid_loss = _scale_loss_values_for_ploting(TrainingMonitor, validator)
    
    BinnedMeanStd, max_bin = _prepare_binned_loss(scaled_train_loss, binsize)
    x_range = np.arange(len(BinnedMeanStd["mean"]))*binsize

    y_mean = np.array(BinnedMeanStd["mean"])
    y1 = y_mean + np.array(BinnedMeanStd["std"])
    y2 = y_mean - np.array(BinnedMeanStd["std"])

    ax_loss_curve.fill_between(x_range, y1, y2, color="navy", alpha=0.2, zorder=1)
    ax_loss_curve.plot(x_range, y_mean, c="navy", zorder=2, label="Training")
    
    x_range_valid = np.arange(len(scaled_valid_loss))*HyperParameters.validation_frequency
    ax_loss_curve.plot(x_range_valid, scaled_valid_loss, c="orange", zorder=2, label="Validation")
    
    ax_loss_curve.set_xlabel("Epochs")
    ax_loss_curve.set_ylabel("{}".format(HyperParameters.loss_function))
    
    ax_loss_curve.legend(loc=1, edgecolor="w",)
#     v.pl.legend(ax_loss_curve, loc=1)
    
    if save_path:
        v.ut.mkdir_flex(save_path)
        img_save_path = os.path.join(save_path, "{}.png".format(TrainingMonitor.current_epoch))
        plt.savefig(img_save_path, bbox_inches="tight")
#     plt.show()
    display(plt.gcf())
    
#     plot_time_stop = time.time() - plot_time_start
#     print("Plotting time: {:.5f}".format(plot_time_stop))
    
class Validator:
    def __init__(
        self,
        network_model,
        diffusion,
        integration_function,
        loss_function,
        TrainingMonitor,
    ):

        self.parallel = True
        self.network_model = network_model
        self.diffusion = diffusion
        self.integration_function = integration_function
        self.loss_function = loss_function
        self.TrainingMonitor = TrainingMonitor
        self.pred_y = {}
        self.y = {}

    def forward_integrate(self, adata, group, use="X", time_key="time"):
        
        self.GroupSizesDict = _get_n_by_training_group(adata)
        adata_group = adata[adata.obs[group]]
            
        self.y[group], self.y0, self.t = _fetch_data(adata_group, use, time_key)

        with torch.no_grad():
            if self.parallel and not self.diffusion:

                self.pred_y[group] = self.integration_function(
                    self.network_model.f, self.y0, self.t
                )

    def calculate_loss(self):

        self.train_loss = self.loss_function(
            self.pred_y['train'], self.y['train'].reshape(self.pred_y['train'].shape)
        ).item()
        self.valid_loss = self.loss_function(
            self.pred_y['valid'], self.y['valid'].reshape(self.pred_y['valid'].shape)
        ).item()
        self.TrainingMonitor.update_loss(self.valid_loss, validation=True)

def _validate(adata, 
              network_model, 
              diffusion, 
              integration_function, 
              HyperParameters, 
              TrainingMonitor,
              plot,
              plot_savepath):

    """"""

    validator = Validator(
        network_model,
        diffusion,
        integration_function,
        HyperParameters.loss_function,
        TrainingMonitor,
    )

    validator.forward_integrate(adata, group="train")
    validator.forward_integrate(adata, group="valid")
    validator.calculate_loss()

    if plot:
        _plot_validation(validator,
                         TrainingMonitor, 
                         HyperParameters,
                         binsize=10, 
                         title_fontsize=16, 
                         save_path=plot_savepath)

    print("Validation loss: {:.4f}".format(validator.valid_loss))