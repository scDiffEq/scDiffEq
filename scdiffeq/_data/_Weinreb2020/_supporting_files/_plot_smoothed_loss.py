
import numpy as np
import pandas as pd
import pyrequisites as pyrex
import vinplots
import matplotlib.pyplot as plt

def _construct_loss_plot():
        
    fig = vinplots.Plot()
    fig.construct(nplots=5, ncols=5, width_ratios=[1, 0.05, 1, 0.05, 1], figsize_width=0.8)
    fig.modify_spines(ax="all", spines_to_delete=['top', 'right'], spines_to_move=['left', 'top'], spines_positioning_amount=20)
    
    fig.AxesDict[0][1].remove()
    fig.AxesDict[0][3].remove()
    fig.AxesDict[0][0].grid(True, alpha=0.2, zorder=0)
    fig.AxesDict[0][2].grid(True, alpha=0.2, zorder=0)
    fig.AxesDict[0][4].grid(True, alpha=0.2, zorder=0)
    
    return fig

def _smooth_test_train(df, smoothing_window):
    
    window_trim = int(smoothing_window / 2)

    X_train = pyrex.smooth(df.training_loss, smoothing_window)[window_trim:-window_trim]
    X_test  = pyrex.smooth(df.test_loss, smoothing_window)[window_trim:-window_trim]
    
    return X_train, X_test

def _plot_smoothed_loss(df, smoothing_window=20, save=False):
    
    """
    Smooth and plot the loss from scDiffEq training. 
    
    Parameters:
    -----------
    df
        type: pandas.DataFrame
    
    smoothing_windows
        type: int
        default: 20
    
    save
        type: bool
        default: False
        
    Returns:
    --------
    None
        Plot is printed and optionally saved. 
    
    Notes:
    ------
    """
        
    fig = _construct_loss_plot()
    
    ax1 = fig.AxesDict[0][0]
    ax2 = fig.AxesDict[0][2]
    ax3 = fig.AxesDict[0][4]
    
    titles = ["Training Loss", "Test Loss", "Normalized Loss"]
    y_label = ["Wasserstein Distance", "Wasserstein Distance", "Normalized Wasserstein Distance"]
    X_losses = [X_train, X_test] = _smooth_test_train(df, smoothing_window)
    loss_values = [df.training_loss, df.test_loss]
    min_norm_loss = df.training_loss.min() / df.training_loss.max()
    colors = ["blue", "red"]
    v_ymin = [df.training_loss.min(), df.test_loss.min(), min_norm_loss]
    v_ymax = [df.training_loss.max(), df.test_loss.max(), 0.9,]
    
    for n, ax in enumerate([ax1, ax2, ax3]):
        if n == 2:
            ax.scatter(range(len(df)), df.training_loss / df.training_loss.max(), c="blue", alpha=0.02)
            ax.scatter(range(len(df)), df.test_loss / df.test_loss.max(), c="red", alpha=0.02)
            ax.plot(X_train / df.training_loss.max(), c='blue', label="Train")
            ax.plot(X_test / df.test_loss.max(), c='red', label="Test")
            ax.legend(edgecolor='w', fontsize=12)
        else:
            ax.scatter(range(len(df)), loss_values[n], c=colors[n], alpha=0.02)
            ax.plot(X_losses[n], c=colors[n])
        ax.set_title(titles[n], fontsize=16)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel(y_label[n], fontsize=14)
        ax.vlines(x=df.training_loss.argmin(), ymin=v_ymin[n], ymax=v_ymax[n], linestyle="--", lw=3, color='black', zorder=6)    