
import numpy as np
import os
import pandas as pd
import sklearn.metrics
import scipy.stats
import warnings


from ..._data._Weinreb2020._RetrieveData import _RetrieveData

warnings.filterwarnings("ignore")

def _retrieve_true_fate_biases(adata):
    
    data = _RetrieveData(adata)
    data.neu_mo_test_set_early()
    
    return data._df["neu_vs_mo_percent"].values

def _calculate_pearsons_rho(true_biases, bias_scores):
    
    r, pval = scipy.stats.pearsonr(true_biases, bias_scores)
    
    return r, pval

def _calculate_AUROC(true_biases, bias_scores):
    
    auroc = sklearn.metrics.roc_auc_score(true_biases > 0.5, bias_scores)
    
    return auroc

def _return_existing_methods_bias_scores(adata, early=True):

    """
    if early = True, returns only the 335 cells. else returns everything for which there is a prediction value >= 0.
    """

    obs_df = adata.obs.copy()
    ground_truth = obs_df.filter(regex="smoothed_groundtruth")
    predictions = obs_df.filter(regex="_predictions")
    pred_df = pd.concat([ground_truth, predictions], axis=1)
    pred_df = pred_df[pred_df >= 0].dropna().reset_index()

    if early:
        data = _RetrieveData(adata)
        data.neu_mo_test_set_early()
        _early_df = data._df.reset_index()[["index"]]
        pred_df = pd.merge(pred_df, _early_df)

    pred_df = pred_df.rename({"index": "cell_idx"}, axis=1)

    return pred_df


def _calculate_accuracy(true_biases, bias_scores, mask=None):
    
    MetricDict = {}
    
    if mask.__class__.__name__ == "ndarray":
        true_biases = true_biases[mask]
        bias_scores = bias_scores[mask]
    
    try:
        auroc = _calculate_AUROC(true_biases, bias_scores)
    except:
        auroc = None
    try:
        r, pval = _calculate_pearsons_rho(true_biases, bias_scores)
    except:
        r, pval = None, None
    
    MetricDict['auroc'] = auroc
    MetricDict['r'] = r
    MetricDict['pval'] = pval
    
    return MetricDict

def _annotate_existing_method_accuracies(accuracy_df, pred_df, true_biases):
        
    mask = np.full(len(pred_df), True).tolist()

    ExistingMethodAccuracy = {}
    for key in pred_df.columns[1:]:
        ExistingMethodAccuracy[key] = {}
        accuracy = _calculate_accuracy(true_biases, pred_df[key])
        acc_masked = _calculate_accuracy(true_biases, pred_df[key], mask=mask)
        for _key in accuracy.keys():
            ExistingMethodAccuracy[key][_key] = accuracy[_key]
            ExistingMethodAccuracy[key]["{}_masked".format(_key)] = acc_masked[_key]

        n_scores = len(mask)
        ExistingMethodAccuracy[key]["fraction_masked"] = 0
        
    df = pd.DataFrame(ExistingMethodAccuracy).T.reset_index()
    df = pd.concat([df, accuracy_df]).set_index('index')
    
    return df
    
def _determine_model_accuracy(adata, X_labels, evaluation_outpath, N):

    true_biases = _retrieve_true_fate_biases(adata)
    pred_df = _return_existing_methods_bias_scores(adata)

    AccuracyDict = {}

    for key, value in X_labels.items():
        AccuracyDict[key] = {}
        Accuracy = _calculate_accuracy(true_biases, value["scores"])
        AccuracyMasked = _calculate_accuracy(
            true_biases, value["scores"], value["mask"]
        )
        for _key in Accuracy.keys():
            AccuracyDict[key][_key] = Accuracy[_key]
        for _key in AccuracyMasked.keys():
            AccuracyDict[key]["{}_masked".format(_key)] = AccuracyMasked[_key]

        n_scores = X_labels[key]["scores"].shape[0]
        AccuracyDict[key]["fraction_masked"] = X_labels[key]["n_masked"] / n_scores

    accuracy_df = pd.DataFrame.from_dict(AccuracyDict).T.reset_index()
    accuracy_df.rename({"index": "epoch"}, axis=1)
    
    accuracy_df = _annotate_existing_method_accuracies(accuracy_df, pred_df, true_biases)
    
    accuracy_df_outpath = os.path.join(evaluation_outpath, "accuracy_df.{}_samples.csv".format(N))
    accuracy_df.to_csv(accuracy_df_outpath)
    
    return accuracy_df
