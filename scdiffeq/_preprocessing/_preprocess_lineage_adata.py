
import cell_fate as cf
import numpy as np
import pandas as pd

def _annotate_lineage_time_group(
    adata, batch_size=45, lineage_key="clone_idx", min_t_occupance=2, verbose=False
):

    GroupedByTimeOccupance = {}
    drop_list = []

    for clone, clone_df in adata.obs.groupby("clone_idx"):
        time = np.sort(clone_df["Time point"].unique())
        if len(time) >= min_t_occupance:
            time_key = "".join(str(int(t)) for t in time)
            if not time_key in GroupedByTimeOccupance.keys():
                if verbose:
                    print(" - adding: {} to keys".format(time_key))
                GroupedByTimeOccupance[time_key] = []
            GroupedByTimeOccupance[time_key].append(clone)
        else:
            drop_list.append(clone)

    df = adata.obs.dropna().reset_index(drop=True)
    tmp = np.zeros(df.shape[0])

    for key, value in GroupedByTimeOccupance.items():
        tmp[df.loc[df["clone_idx"].isin(value)].index] = key

    adata.obs["time_group"] = ["-".join([z for z in str(int(j))]) for j in tmp]
    keep_idx = adata.obs.loc[~adata.obs["clone_idx"].isin(drop_list)].index

    return adata[keep_idx].copy()


def _report_lineage_time_occupance(
    filtered_clonal_idx, n_lineages, perc_lineages_occ_t
):
    msg = "{}/{} ({:.2f}%) lineages present at specified time occupance."
    print(msg.format(len(filtered_clonal_idx), n_lineages, perc_lineages_occ_t))


def _filter_by_time_occupance(adata, lineage_key, time_key, n_timepoints, silent):
    """remove clones not present at all timepoints"""

    obs_grouped_by_lineage = adata.obs.groupby(lineage_key)
    n_lineages = len(obs_grouped_by_lineage)

    df = pd.DataFrame(obs_grouped_by_lineage[time_key].nunique() >= n_timepoints)
    filtered_idx = np.array(df[df[time_key] == True].index).astype(int)

    perc_lineages_occ_t = (len(filtered_idx) / n_lineages) * 100

    adata_filtered = adata[
        adata.obs.loc[adata.obs[lineage_key].isin(filtered_idx)].index.astype(int)
    ]
    if not silent:
        _report_lineage_time_occupance(filtered_idx, n_lineages, perc_lineages_occ_t)

    return adata_filtered


def _augment_time(
    adata, time_key="Time point", key_added="t", TimeConversion={2: 0, 4: 0.01, 6: 0.02}
):

    df = adata.obs.copy()

    if key_added in df.columns:
        df = df.drop(key_added, axis=1)

    time_df = pd.DataFrame.from_dict(
        TimeConversion, orient="index"
    ).reset_index()  # , index=range(len(TimeConversion)))
    time_df.columns = [time_key, key_added]
    time_df.index = time_df.index.astype(str)
    df.index = df.index.astype(str)

    adata.obs = df.merge(time_df, on=time_key, how="left")


class PreprocessLineages:
    def __init__(
        self, adata, lineage_key="clone_idx", time_key="Time point", silent=False
    ):

        """Preprocess lineage data for benchmark"""

        self._adata = adata
        self._adata.obs.index = self._adata.obs.index.astype(str)
        self._adata.var.index = self._adata.var.index.astype(str)
        cf.pp.annotate_train_test(self._adata)
        cf.pp.annnotate_unique_test_train_lineages(self._adata)
        self._lineage_key = lineage_key
        self._time_key = time_key
        self._silent = silent
        self._n_timepoints = adata.obs[time_key].nunique()

    def filter_on_time_occupance(self, n_timepoints=False):

        if not n_timepoints:
            nt = self._n_timepoints
        else:
            nt = n_timepoints

        self._adata = _filter_by_time_occupance(
            self._adata,
            self._lineage_key,
            self._time_key,
            self._n_timepoints,
            self._silent,
        )

    def augment_time(self, TimeConversion={2: 0, 4: 0.01, 6: 0.02}):

        self._adata.obs.index = self._adata.obs.index.astype(str)
        self._adata.var.index = self._adata.var.index.astype(str)

        _augment_time(
            self._adata, time_key=self._time_key, TimeConversion=TimeConversion
        )
        
    def annotate_lineage_time_group(self, batch_size):
        
        self._adata.obs.index = self._adata.obs.index.astype(str)
        self._adata.var.index = self._adata.var.index.astype(str)
        
        _annotate_lineage_time_group(self._adata,
                                     batch_size=batch_size,
                                     lineage_key=self._lineage_key,
                                     min_t_occupance=self._n_timepoints,
                                     verbose=False,
                                    )

    def training_data(self):

        self._adata.obs.index = self._adata.obs.index.astype(str)
        self._adata.var.index = self._adata.var.index.astype(str)

        self._adata_train = self._adata[self._adata.obs[self._adata.obs["train"]].index]
        return self._adata_train

    def test_data(self):
        self._adata.obs.index = self._adata.obs.index.astype(str)
        self._adata.var.index = self._adata.var.index.astype(str)

        self._adata_test = self._adata[self._adata.obs[self._adata.obs["test"]].index]
        return self._adata_test


def _preprocess_lineage_adata(adata, batch_size=45, benchmark=True):

    lineage_pp = PreprocessLineages(adata)
    lineage_pp.filter_on_time_occupance()
    lineage_pp.augment_time()
    lineage_pp.annotate_lineage_time_group(batch_size)

    if benchmark:
        return lineage_pp.training_data(), lineage_pp.test_data()

    else:
        return lineage_pp._adata