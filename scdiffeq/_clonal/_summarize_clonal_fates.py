
import pandas as pd
from tqdm.notebook import tqdm
import vinplots
import warnings

warnings.filterwarnings("ignore")

from ._count_clones import _count_clones
from ._count_annotations import _count_annotations
from ._count_clones_in_annotation import _count_clones_in_annotation

def _get_n_cells_per_clone_by_annotation(clone_df, fate):

    celltype_df = clone_df.loc[clone_df["top_celltype"] == fate]
    celltype_df["n_clone_celltype"] = (
        (celltype_df["max_proportion"] * celltype_df["sum"]).round().astype(int)
    )
    celltype_df = celltype_df.sort_values("n_clone_celltype", ascending=False)

    x = celltype_df.index
    y = celltype_df["n_clone_celltype"]

    return x, y

def _count_annotations_single_clone(adata, clone_id, annotation_key, clone_id_key):

    obs_df = adata.obs.copy()
    counted_clone = (
        obs_df.loc[obs_df[clone_id_key] == clone_id]
        .groupby(annotation_key)
        .count()
        .rename({clone_id_key: int(clone_id)}, axis=1)[clone_id]
    )

    return counted_clone


def _get_top_celltype(clone_count_df):

    return [
        clone_count_df.columns[clone_count_df.iloc[row].argmax()]
        for row in range(len(clone_count_df))
    ]


def _annotate_intraclone_populations(clone_count_df, unique_clones):

    clone_max = clone_count_df.max(axis=1)
    clone_sum = clone_count_df.sum(axis=1)

    clone_count_df["top_celltype"] = _get_top_celltype(clone_count_df)
    clone_count_df["sum"] = clone_sum
    clone_count_df["max_proportion"] = round(clone_max / clone_sum, ndigits=3)
    clone_count_df["n_clone_celltype"] = (
        (clone_count_df["max_proportion"] * clone_count_df["sum"]).round().astype(int)
    )
    clone_count_df["clone_idx"] = unique_clones
    clone_count_df = clone_count_df.sort_values(
        "n_clone_celltype", ascending=False
    ).reset_index(drop=True)
    
    return clone_count_df


def _count_clones_in_annotation(
    adata, annotation_key="Annotation", clone_id_key="clone_idx"
):

    obs_df = adata.obs.dropna()
    unique_clones = obs_df[clone_id_key].unique()
    counted_clone_list = []

    for n, clone_id in tqdm(enumerate(unique_clones)):
        counted_clones = _count_annotations_single_clone(
            adata, clone_id, annotation_key, clone_id_key
        )
        counted_clone_list.append(counted_clones)

    clone_count_df = pd.concat(counted_clone_list, axis=1).T
    clone_count_df = _annotate_intraclone_populations(clone_count_df, unique_clones)
    clone_count_df.columns.name = None

    return clone_count_df


def _linearize_axes_dict(AxesDict):

    axes = []
    for i in AxesDict.keys():
        for j in AxesDict[i]:
            axes.append(AxesDict[i][j])

    return axes


def _build_clonal_fate_summary_plot():

    fig = vinplots.Plot()
    fig.construct(nplots=11, ncols=4, figsize=1, hspace=0.4)
    fig.modify_spines(ax="all", spines_to_delete=["top", "right"])
    axes = _linearize_axes_dict(fig.AxesDict)

    return fig, axes


def _plot_dominant_clonal_fate_one_ax(ax, clone_df, fate, color):

    x, y = _get_n_cells_per_clone_by_annotation(clone_df, fate)
    ax.scatter(x, y, c=color)
    ax.set_title("{} clones\n(n={})".format(fate, len(clone_df)))


def _plot_dominant_clonal_fate_summary(clone_df, counted_annotations):

    fig, axes = _build_clonal_fate_summary_plot()
    colors = vinplots.palettes.SHAREseq()[:10] + ["lightgrey"]

    for n, fate in enumerate(counted_annotations.index):
        if fate == "undiff":
            continue
        else:
            _plot_dominant_clonal_fate_one_ax(
                axes[n - 1], clone_df, fate, color=colors[n - 1]
            )

        _plot_dominant_clonal_fate_one_ax(axes[n], clone_df, "undiff", color=colors[n])


def _summarize_clonal_fates(adata, clone_df=None):

    counted_annotations = _count_annotations(adata, annotation_key="Annotation")

    try:
        clone_df
    except:
        clone_df = _count_clones_in_annotation(
            adata, annotation_key="Annotation", clone_id_key="clone_idx"
        )

    _plot_dominant_clonal_fate_summary(clone_df, counted_annotations)

    return clone_df, counted_annotations