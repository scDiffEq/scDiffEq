
# package imports #
# --------------- #
import vintools as v
import matplotlib.pyplot as plt

def _simulation_plot_presets(
    x, y, plot_title="Simulation", x_lab="$x$", y_lab="$y$", title_fontsize=14, label_fontsize=12, figsize=(6, 5), **kwargs
):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    v.pl.delete_spines(ax)

    ax.get_xaxis().set_ticks([x.min().round(), x.max().round()])
    ax.get_yaxis().set_ticks([y.min().round(), y.max().round()])
    ax.set_title(plot_title, y=1.05, fontsize=title_fontsize)
    ax.set_xlabel("$x$", size=label_fontsize)
    ax.set_ylabel("$y$", size=label_fontsize)
    ax.scatter(x, y, **kwargs)
    
def _plot(self, c='time', **kwargs):
    
    X = self.adata.X
    if c=='time':
        c = self.adata.obs.time
        
    _simulation_plot_presets(X[:,0], X[:,1], c=c, **kwargs)