
import numpy as np
import pandas as pd


def _create_lineage_dict(adata, lineage_key="clone_idx", time_key="Time point"):

    """

    adata

    lineage_key
        default: "clone_idx"

    time_key
        default: "Time point"

    """

    LineageDict = {}

    for clone, clone_df in adata.obs.groupby(lineage_key):
        LineageDict[clone] = {}

        unique_t = np.sort(clone_df[time_key].unique())

        LineageDict[clone]["unique_t"] = unique_t
        LineageDict[clone]["cell_df"] = clone_df

    return LineageDict


def _annotate_lineage_resolved_time(
    adata, lineage_key="clone_idx", time_key="Time point"
):

    """
    Create a Lineage Dict containing info about the time scale / dispersion
    of each trajectory then add this to AnnData.

    Parameters:
    -----------
    adata

    lineage_key
        default: "clone_idx"

    time_key
        default: "Time point"
    
    Returns:
    --------
    
    """

    LineageDict = _create_lineage_dict(adata, lineage_key, time_key)

    lineage_type = []
    lineage_dt = []
    lineage_nt = []

    for lineage in LineageDict.keys():

        _t = LineageDict[lineage]["unique_t"]
        _dt = _t.max() - _t.min()

        if len(_t) == 3:
            lineage_type.append("{}_{}_{}".format(_t[0], _t[1], _t[2]))
        else:
            lineage_type.append("{}_{}".format(_t.min(), _t.max()))

        lineage_nt.append(len(_t))
        lineage_dt.append(_dt)

    lineage_df = pd.DataFrame(
        [LineageDict.keys(), lineage_nt, lineage_dt, lineage_type]
    ).T
    lineage_df.columns = [lineage_key, "nt", "dt", "dt_type"]

    # filter single-timepoint lineages
    lineage_df_filt = lineage_df.loc[lineage_df["nt"] > 1]

    # merge with the remaining adata.obs
    lineage_df_filt.reset_index(drop=True)
    obs_df = adata.obs.merge(lineage_df_filt, on=lineage_key, how="outer")
    obs_df.index = obs_df.index.astype(str)
    adata.obs = obs_df

    return adata