
# package imports #
# --------------- #
import matplotlib.pyplot as plt

def _remove_spines(ax, spines=["top", "right", "bottom", "left"]):

    "https://newbedev.com/how-to-remove-frame-from-matplotlib-pyplot-figure-vs-matplotlib-figure-frameon-false-problematic-in-matplotlib"

    for spine in spines:
        ax.spines[spine].set_visible(False)


def _simulation_plot_presets(
    x, y, title_fontsize=14, label_fontsize=12, figsize=(6, 5), **kwargs
):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    _remove_spines(ax)

    ax.get_xaxis().set_ticks([x.min().round(), x.max().round()])
    ax.get_yaxis().set_ticks([y.min().round(), y.max().round()])
    ax.set_title("Simulation", y=1.05, fontsize=title_fontsize)
    ax.set_xlabel("$x$", size=label_fontsize)
    ax.set_ylabel("$y$", size=label_fontsize)
    ax.scatter(x, y, **kwargs)
    
def _plot(self, c='time', **kwargs):
    
    X = self.adata.X
    if c=='time':
        c = self.adata.obs.time
        
    _simulation_plot_presets(X[:,0], X[:,1], c=c, **kwargs)