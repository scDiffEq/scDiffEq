
import matplotlib.pyplot as plt
import os
import pandas as pd
import vinplots


def _build_loss_plot(nplots):

    fig = vinplots.Plot()
    fig.construct(nplots=nplots, ncols=nplots, figsize=1.2)
    fig.modify_spines(ax="all", spines_to_delete=["top", "right"])

    axes = [fig.AxesDict[0][i] for i in range(nplots)]

    return fig, axes


def _plot_saved_status(ax, df, x_key="epoch", y_key="total", title=None):

    ax.plot(df[x_key], df[y_key], marker="o", c="darkred")
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_xlabel(y_key)

def _read_saved_status_log(RunInfo, plot=True):
    
    """
    Check the status of a run based on what's written to the log file.
    
    TO-DO: Add distinct plots for test and then train + test in the same plot. one row.
    
    
    """

    log_path = os.path.join(RunInfo.run_outdir, "status.log")
    log_df = pd.read_csv(log_path, sep="\t")

    grouped_log_df = log_df.groupby("mode")

    if plot:
        fig, axes = _build_loss_plot(nplots=len(grouped_log_df))

        for n, (mode, mode_df) in enumerate(grouped_log_df):
            plot_title = "{}: epoch: {}".format(mode, len(mode_df))
            _plot_saved_status(axes[n], mode_df, x_key="epoch", y_key="total", title=plot_title)
        plt.show()

    return log_df