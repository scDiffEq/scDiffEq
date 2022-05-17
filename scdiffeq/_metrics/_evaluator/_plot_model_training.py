
import licorice_font as font
import matplotlib.pyplot as plt
import numpy as np
import os
import vinplots


from ._parse_results_functions import _return_epochs_to_evaluate

def _make_savename(save_prefix, savename):
    
    if not type(savename) == str:
        savename = ""
        
    if not (
        savename.endswith(".png")
        or savename.endswith(".svg")
        or savename.endswith(".pdf")
    ):
        savename = savename + ".svg"
    
    if savename == ".svg":
        return "".join([save_prefix, savename])
    else:
        return ".".join([save_prefix, savename])
        
def _save_fig(savename, outpath, layers, nodes, seed):
    
    save_prefix = "scDiffEq.training.{}layers.{}nodes.seed{}".format(layers, nodes, seed)
    save_prefix = os.path.join(outpath, save_prefix)
    
    if savename:
        savename = _make_savename(save_prefix, savename)
        print("\n{}: {}\n".format(font.font_format("Saving training plot to", ["BOLD"]), savename))
        plt.savefig(savename)

def _mkfig(figsize=1.5):
    
    fig = vinplots.Plot()
    fig.construct(nplots=1, ncols=1, figsize=figsize)
    fig.modify_spines(ax="all", spines_to_delete=['top', 'right'])
    ax = fig.AxesDict[0][0]
    
    return fig, ax

def _plot_training(log_df, savename, outpath, layers, nodes, seed, figsize=1.5):
    
    fig, ax = _mkfig(figsize)
    
    epochs_to_evaluate = _return_epochs_to_evaluate(log_df)
    y_max = max(log_df['total'].max(), 500)
    
    ax.fill_between(x=log_df['epoch'], y2=log_df['d2'], y1=log_df['d4'], color="#81C2A5", alpha=0.25)
    ax.fill_between(x=log_df['epoch'], y2=log_df['d4'], y1=log_df['d4'] + log_df['d6'], color="#537E70", alpha=1)
    ax.scatter(log_df['epoch'], log_df['total'], c="#314240", s=2)
    ax.set_ylim(0, y_max)
    ax.set_title("Training Progress")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Wasserstein Distance")
    ax.vlines(x=epochs_to_evaluate, ymin=0, ymax=y_max, color="black", lw=2, ls="--")
    
    _save_fig(savename, outpath, layers, nodes, seed)
    plt.show()
    