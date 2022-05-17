
import sklearn.metrics

from ._metrics_utilities._load_saved_true_bias_scores import _load_saved_true_bias_scores


def _AUROC(bias_scores, true_biases=False):

    if not true_biases:
        true_biases = _load_saved_true_bias_scores()

    auroc = sklearn.metrics.roc_auc_score(true_biases > 0.5, bias_scores)

    return auroc