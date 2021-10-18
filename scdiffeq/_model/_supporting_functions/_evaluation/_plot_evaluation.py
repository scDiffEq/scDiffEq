
import matplotlib.pyplot as plt
import vintools as v
import numpy as np

def _format_test_predictions_for_plotting(evaluator):

    x, y = evaluator.y.detach().numpy()[:, 0].T, evaluator.y.detach().numpy()[:, 1].T
    x_, y_ = (
        evaluator.pred_y.detach().numpy()[:, :, 0].T,
        evaluator.pred_y.detach().numpy()[:, :, 1].T,
    )

    return [x, y, x_, y_]


def _tile_time(time, test_pred):
    return np.tile(time, test_pred.shape[1])


def _plot_evaluation(evaluator, title_fontsize=12):

    plot = v.pl.ScatterPlot()
    plot.construct_layout(nplots=3)
    plot.style()

    ax_test = plot.AxesDict[0][0]
    ax_pred = plot.AxesDict[0][1]
    ax_over = plot.AxesDict[0][2]

    [x, y, x_, y_] = _format_test_predictions_for_plotting(evaluator)
    t = _tile_time(evaluator.t, evaluator.pred_y)

    ax_test.scatter(x, y, c=t, zorder=1)
    ax_pred.scatter(x_, y_, c=t, zorder=1)
    ax_over.scatter(x, y, c="lightgrey", alpha=0.75, zorder=1)
    ax_over.scatter(x_, y_, c=t, zorder=2)

    ax_test.set_title("Test Data", fontsize=title_fontsize)
    ax_pred.set_title("Test Predictions", fontsize=title_fontsize)
    ax_over.set_title("Prediction Overlay", fontsize=title_fontsize)

    plt.show()