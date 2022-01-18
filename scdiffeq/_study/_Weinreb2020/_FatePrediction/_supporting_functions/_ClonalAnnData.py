
import anndata as a
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def _loc_timepoint(df_clone, timepoint):
    return df_clone.loc[df_clone["Time point"] == timepoint].index.astype(int)


def _get_all_three_point_lineages_one_clone(TimepointData):

    indices = []

    for idx_2 in TimepointData[2]:
        for idx_4 in TimepointData[4]:
            for idx_6 in TimepointData[6]:
                indices.append([idx_2, idx_4, idx_6])
    return indices


def _get_threepoint_lineage_cells(obs_df):

    """"""

    grouped_obs_df = obs_df.groupby(["clone_idx"])
    grouped_df_3pts = (grouped_obs_df["Time point"].nunique() == 3).reset_index()
    clone_idx = grouped_df_3pts.loc[grouped_df_3pts["Time point"] == True][
        "clone_idx"
    ].values.astype(int)

    obs_df_filtered_3pts = obs_df.loc[obs_df["clone_idx"].isin(clone_idx)]

    return obs_df_filtered_3pts


def _aggregate_clonal_lineages(CloneIndexDict):

    clonal_lineage_indices = []
    for lin in CloneIndexDict.values():
        clonal_lineage_indices = clonal_lineage_indices + lin

    return clonal_lineage_indices


def _get_clone_indices_by_timepoint(df_obs):

    clones = _get_threepoint_lineage_cells(df_obs)["clone_idx"].dropna().unique()
    timepoints = np.sort(df_obs["Time point"].unique())

    CloneIndexDict = {}
    for clone in clones:
        df_clone = df_obs.loc[df_obs["clone_idx"] == clone]
        TimepointData = {}
        for timepoint in timepoints:
            TimepointData[timepoint] = _loc_timepoint(df_clone, timepoint)
        CloneIndexDict[clone] = _get_all_three_point_lineages_one_clone(TimepointData)
    #         CloneIndexDict[clone] = TimepointData

    clonal_lineage_idxs = _aggregate_clonal_lineages(CloneIndexDict)

    return clonal_lineage_idxs


def _multiloc(df, columns=[], values=[]):

    """Subset a pandas DataFrame by [multiple] label columns and corresponding vlaues."""

    df_ = df.copy()

    for n, column in enumerate(columns):
        df_ = df_.loc[df_[column] == values[n]]

    return df_


def _format_lineage_data_by_time(DataDict, timepoints):

    """"""

    n_genes = DataDict["X_lineages"].shape[-1]
    n_pcs = DataDict["X_pca_lineages"].shape[-1]
    n_umap_components = DataDict["X_umap_lineages"].shape[-1]

    FormattedData = {}
    for key in DataDict.keys():
        FormattedData[key] = {}
        data_to_reshape = []
        FormattedData[key]["time_counts"] = {}
        for i in range(len(timepoints)):
            FormattedData[key]["time_counts"][timepoints[i]] = len(DataDict[key])
            data_to_reshape.append(DataDict[key][:, i, :])
        FormattedData[key]["data"] = np.vstack(data_to_reshape)

    return FormattedData


def _organize_data_by_clonal_lineages_and_timepoint(adata, clonal_lineage_idxs):

    DataDict = {}
    DataDict["X_lineages"] = adata.X.toarray()[clonal_lineage_idxs]
    DataDict["X_pca_lineages"] = adata.obsm["X_pca"][clonal_lineage_idxs]
    DataDict["X_umap_lineages"] = adata.obsm["X_umap"][clonal_lineage_idxs]

    timepoints = np.sort(adata.obs["Time point"].unique())

    FormattedData = _format_lineage_data_by_time(DataDict, timepoints)

    return FormattedData


def _create_simple_adata(FormattedData, clonal_lineage_idxs):

    t_arranged = np.array([])
    for k, v in FormattedData["X_lineages"]["time_counts"].items():
        t_arranged = np.append(t_arranged, np.full(v, k))

    simple_adata = a.AnnData(FormattedData["X_lineages"]["data"])
    simple_adata.obs["t"] = t_arranged
    simple_adata.obsm["X_pca"] = FormattedData["X_pca_lineages"]["data"]
    simple_adata.obsm["X_umap"] = FormattedData["X_umap_lineages"]["data"]
    simple_adata.obs["clone"] = np.tile(range(len(clonal_lineage_idxs)), 3)

    return simple_adata


def _formulate_lineages_for_neural_diffeq_training(adata):

    """"""

    df_obs = _multiloc(adata.obs, ["neu_mo_mask"], [True])
    clonal_lineage_idxs = _get_clone_indices_by_timepoint(df_obs)
    FormattedData = _organize_data_by_clonal_lineages_and_timepoint(
        adata, clonal_lineage_idxs
    )

    simple_adata = _create_simple_adata(FormattedData, clonal_lineage_idxs)

    return simple_adata

def _add_incomplete_obs_annotation(adata, idx, array, key_added):

    """Add an annotation to adata.obs that only applys to a subset of cells"""

    empty_col = np.zeros(len(adata))
    empty_col[idx] = array
    adata.obs[key_added] = empty_col

    return adata


def _load_early_NeuMon_predictions_Weinreb2020(adata, base_dir):

    adata.obs["timepoints"] = np.load(base_dir + "timepoints.npy")
    adata.obs["neu_mo_mask"] = np.load(
        base_dir + "neutrophil_monocyte_trajectory_mask.npy"
    )

    df_obs = adata.obs.copy()

    early_NeuMon_idx = (
        df_obs.loc[df_obs["neu_mo_mask"] == True]
        .loc[df_obs["timepoints"] == 2]
        .index.astype(int)
    )

    return early_NeuMon_idx


def _load_prediciton_method_results(
    adata,
    base_dir,
    methods=[
        "smoothed_groundtruth_from_heldout",
        "PBA",
        "FateID",
        "WOT",
    ],
    plot=False,
):

    early_NeuMon_idx = _load_early_NeuMon_predictions_Weinreb2020(adata, base_dir)

    for n, method in enumerate(methods):
        empty_col = np.ones(len(adata)) * -1
        filepath = glob.glob(os.path.join(base_dir, method + "*.npy"))[0]
        empty_col[early_NeuMon_idx] = np.load(filepath)
        adata.obs[method] = empty_col
        if plot:
            plt.hist(empty_col[idx], bins=100, alpha=1 - 0.2 * n)

    return adata, early_NeuMon_idx


def _load_clonal_annotation_files(adata, base_dir, methods, plot):

    adata.uns["clonal_fate_matrix"] = np.load(base_dir + "clonal_fate_matrix.npy")
    adata, early_NeuMon_idx = _load_prediciton_method_results(
        adata, base_dir, methods, plot
    )
    adata = _add_incomplete_obs_annotation(
        adata,
        idx=np.where(adata.obs["Time point"] == 2),
        array=np.load(base_dir + "early_cells.npy"),
        key_added="early_cells",
    )
    adata = _add_incomplete_obs_annotation(
        adata,
        idx=early_NeuMon_idx,
        array=np.load(base_dir + "heldout_mask.npy")
        + 1,  # results in three possible values: [0, 1, 2] where 2 == True, 1 == False
        key_added="heldout_mask",
    )

    return adata

class _ClonalAnnData:
    def __init__(self, adata, annot_dir="./"):

        """Class to load and organize the clonal annotation files from Weinreb2020"""

        self._adata = adata
        self._annot_dir = annot_dir

    def load(
        self,
        methods_and_masks=["smoothed_groundtruth_from_heldout", "PBA", "FateID", "WOT"],
        plot=False,
        return_adata=True,
    ):

        """"""

        self._adata = _load_clonal_annotation_files(
            self._adata, self._annot_dir, methods_and_masks, plot
        )
        if return_adata:
            return self._adata.copy()
    
    def prepare_for_training(self, simple_adata_path="simple.adata.input.h5ad"):
        
        """"""
        self.simple_adata = _formulate_lineages_for_neural_diffeq_training(self._adata)
        self.simple_adata.write_h5ad(simple_adata_path)