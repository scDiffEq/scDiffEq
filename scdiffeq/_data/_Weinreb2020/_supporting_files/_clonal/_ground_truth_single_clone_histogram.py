import numpy as np
import pandas as pd
import vinplots

from ._subset_single_clonal_lineage import _subset_single_clonal_lineage

def _return_validated_timepoint_celltypes(
    t,
    annotations,
    validated_cols,
):

    return pd.DataFrame(
        data=np.array(
            [
                np.repeat(t, len(annotations)),
                np.tile(annotations, len(t)),
            ]
        ).T,
        columns=validated_cols,
    )


def _validate_plottable_counted_df(
    df_counted,
    t=[2, 4, 6],
    annotations=["undiff", "Monocyte", "Neutrophil"],
    validated_cols=["Time point", "Annotation"],
    numerical_cols=["Time point"],
):

    """Check if counted DataFrame has all days / celltypes included, even if only zeros."""

    df_empty = _return_validated_timepoint_celltypes(
        t,
        annotations,
        validated_cols,
    )

    for col in numerical_cols:
        df_empty[col] = df_empty[col].astype(int)
        df_counted[col] = df_counted[col].astype(int)

    df_validated = (
        pd.merge(df_empty, df_counted, how="left", on=validated_cols)
        .sort_values(validated_cols, ascending=(True, False))
        .fillna(0)
    )

    return df_validated


def _format_obs_to_celltype_counts(
    clone_adata,
    time_key="Time point",
    annotation_key="Annotation",
    celltypes=["Monocyte", "Neutrophil", "undiff"],
    t=[2, 4, 6],
    validated_cols=["Time point", "Annotation"],
    numerical_cols=["Time point"],
):

    df = clone_adata.obs.copy()
    df = (
        df.groupby([time_key, annotation_key])
        .count()
        .reset_index()[[time_key, annotation_key, "clone_idx"]]
    )
    df = df.loc[df[annotation_key].isin(celltypes)].sort_values(
        [time_key, annotation_key], ascending=(True, False)
    )

    df_validated = _validate_plottable_counted_df(
        df, t, celltypes, validated_cols, numerical_cols
    )

    return df_validated


def _plot_d6_NeuMon_clonal_cell_population_histogram(
    ax,
    clone_adata,
    xaxis_offset=[-0.5, 0, 0.5],
    w=0.45,
    colorscheme=[
        "#F0EFEB",
        "#112F45",
        "#E28C32",
    ],
    plot=False,
    **kwargs
):

    df_counts = _format_obs_to_celltype_counts(clone_adata, **kwargs)
    
    if plot:
        labels = [
            "Undiff.",
            "Neutrophil\nDay 2",
            "Monocyte",
            "Undiff.",
            "Neutrophil\nDay 4",
            "Monocyte",
            "Undiff.",
            "Neutrophil\nDay 6",
            "Monocyte",
        ]

        ax.grid(True, zorder=0, alpha=0.2)

        x_ = np.arange(0, 6, 2)
        y_ = df_counts["clone_idx"]

        x_pos = np.array([])
        for n, celltype in enumerate(["undiff", "Neutrophil", "Monocyte"]):
            x_pos = np.append(x_pos, x_ + xaxis_offset[n])
            ax.bar(
                x_ + xaxis_offset[n],
                df_counts.loc[df_counts["Annotation"] == celltype]["clone_idx"],
                width=w,
                zorder=2,
                color=colorscheme[n],
                edgecolor="grey",
            )

        ax.set_xticks(np.sort(x_pos))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Clone Count", fontsize=16)

        clone = clone_adata.obs.clone_idx.unique().astype(int).astype(str)[0]

        ax.set_title("Clone: {}".format(clone), fontsize=20)

    return df_counts


def _build_ground_truth_single_clone_histogram():

    fig = vinplots.Plot()
    fig.construct(nplots=1, ncols=1, figsize=2)
    fig.modify_spines(
        ax="all",
        spines_to_delete=["top", "right"],
        spines_to_move=["bottom", "left"],
        spines_positioning_amount=15,
        spines_to_color=["bottom", "left"],
        color="grey",
    )
    ax = fig.AxesDict[0][0]

    return fig, ax


def _ground_truth_single_clone_histogram(
    clone_adata=False, clone_idx=False, adata=False, plot=False, **kwargs
):

    if not clone_adata:
        clone_adata = _subset_single_clonal_lineage(adata, clone_idx)
    if plot:
        fig, ax = _build_ground_truth_single_clone_histogram()
    else:
        fig, ax = None, None

    df_clone = _plot_d6_NeuMon_clonal_cell_population_histogram(
        ax, clone_adata, plot, **kwargs
    )

    return df_clone