
import pandas as pd
from tqdm.notebook import tqdm
import vinplots
import warnings

warnings.filterwarnings("ignore")


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