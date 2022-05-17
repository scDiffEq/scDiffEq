
import scipy.stats

from ._metrics_utilities._load_saved_true_bias_scores import _load_saved_true_bias_scores

def _pearsons_rho(bias_scores, true_biases=False):

    if not true_biases:
        true_biases = _load_saved_true_bias_scores()
        
    r, pval = scipy.stats.pearsonr(true_biases, bias_scores)

    return r, pval