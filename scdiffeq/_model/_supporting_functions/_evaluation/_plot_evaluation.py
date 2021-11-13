
import matplotlib.pyplot as plt
import numpy as np
import os

def _format_test_predictions_for_plotting(pred_y, y):

    x, y = y.detach().cpu().numpy()[:, 0].T, y.detach().cpu().numpy()[:, 1].T
    x_, y_ = (
        pred_y.detach().cpu().numpy()[:, :, 0],
        pred_y.detach().cpu().numpy()[:, :, 1],
    )

    return [x, y, x_, y_]


def _tile_time(time, test_pred):
    return np.tile(time.detach().cpu().numpy(), test_pred.shape[0])

# def _plot_evaluation(evaluator, title_fontsize=16, save_path=False, TrainingMonitor=False):

#     plot = v.pl.ScatterPlot()
#     plot.construct_layout(nplots=3)
#     plot.style()

#     ax_test = plot.AxesDict[0][0]
#     ax_pred = plot.AxesDict[0][1]
#     ax_over = plot.AxesDict[0][2]
    
#     for batch in evaluator.BatchedPredictions.keys():
#         pred_y = evaluator.BatchedPredictions[batch]
#         y = evaluator.BatchedData[batch]["y"]
#         t = evaluator.BatchedData[batch]["t"]
        

#         [x, y, x_, y_] = _format_test_predictions_for_plotting(pred_y, y)
#         t = _tile_time(t, pred_y)

#         ax_test.scatter(x, y, c=t, zorder=10)
#         ax_pred.scatter(x_, y_, c=t, zorder=10)
#         ax_over.scatter(x, y, c="lightgrey", alpha=0.75, zorder=1)
#         ax_over.scatter(x_, y_, c=t, zorder=10)

#     ax_test.set_title("Test Data", fontsize=title_fontsize)
#     ax_pred.set_title("Test Predictions", fontsize=title_fontsize)
#     ax_over.set_title("Prediction Overlaid", fontsize=title_fontsize)

#     if save_path:
#         v.ut.mkdir_flex(save_path)
#         img_save_path = os.path.join(save_path, "evaluation.{}.png".format(TrainingMonitor.current_epoch))
#         plt.savefig(img_save_path, bbox_inches="tight")
#     plt.show()
