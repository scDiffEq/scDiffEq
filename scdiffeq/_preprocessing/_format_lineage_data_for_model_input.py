
import numpy as np
import torch

from .._data._Weinreb2020._TestSet import _TestSet

def _cells_at_time_final(time_grouped_lineage):

    "returns the number of cells at time final"

    time_final = np.array(list(time_grouped_lineage.groups.keys())).max()
    n_samples = time_grouped_lineage.get_group(time_final).shape[0]
    return n_samples


def _sample_data_at_t(adata, time_df, n_samples, use_key):

    _idx = time_df.index
    _idx = np.random.choice(_idx, n_samples)
    _xt = torch.Tensor(adata[_idx].obsm[use_key])

    return _xt


def _format_lineage_data_for_model_input(
    adata,
    use_key="X_pca",
    lineage_key="clone_idx",
    groupby="dt_type",
    time_key="Time point",
):

    """
    groupby (dt_type_key):
        The groupby key to adata.obs[''] containing the dt type. critical for compatible forward integration.
        default: "dt_type"

    """
    
    test = _TestSet(adata)

    test_adata = test.adata
    test_lineages = test.lineages

    grouped = adata.obs.groupby(groupby)
    FormattedData = {}

    for dt_type, dt_group in grouped:
        if not dt_group["nt"].unique()[0] == 1:
            FormattedData[dt_type] = {}
            original_t = np.sort(dt_group[time_key].unique())
            t = torch.Tensor(np.sort(dt_group["t"].unique()))
            _dt_type = dt_group[groupby].str.split("_", expand=True)
            lineages = dt_group[lineage_key].unique()
            shuffled_lineages = np.random.choice(
                lineages, len(lineages), replace=False
            )  # only matters for batch SGD
            X_test = []
            X_train = []
            X_train_idx = []
            X_test_idx = []
            
            for lineage in shuffled_lineages:
                time_grouped_lineage = adata.uns["LineageDict"][lineage][
                    "cell_df"
                ].groupby(time_key)
                n_samples = _cells_at_time_final(time_grouped_lineage)
                
                _X_lineages = []
                for _t, _t_df in time_grouped_lineage:
                    _xt = _sample_data_at_t(
                        adata, _t_df, n_samples, use_key
                    )
                    _X_lineages.append(_xt)

                if lineage in test_lineages:
                    X_test.append(torch.stack(_X_lineages))
                    X_test_idx.append(n_samples)
                else:
                    X_train.append(torch.stack(_X_lineages))
                    X_train_idx.append(n_samples)
              
            keys   = ["t", "X_train", "X_train_idx", "X_test", "X_test_idx"]
            values = [t, X_train, X_train_idx, X_test, X_test_idx]
            
            for _key, _value in zip(keys, values):
                if ("X_" in _key) & (not "_idx" in _key):
                    try:
                        FormattedData[dt_type][_key] = torch.hstack(_value)
                    except:
                        FormattedData[dt_type][_key] = None
                elif ("X_" in _key) & ("_idx" in _key):
                    FormattedData[dt_type][_key] = np.append(0, np.array(_value)).cumsum()
                elif _key == "t":
                    FormattedData[dt_type][_key] = torch.Tensor(_value)
                else:
                    FormattedData[dt_type][_key] = None
                    print("{}\t| {} did not work...".format(dt_type, _key))
        
        print(" - Formatted data with time-points:", end=" ")
        for n, _time in enumerate(original_t):
            if n+1 != len(t):
                print("{}".format(_time), end=", ")
            else:
                print("{}".format(_time))
            
    return FormattedData
