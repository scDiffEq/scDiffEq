import vinplots


def _simple_scatter(figsize=1, nplots=1):

    fig = vinplots.Plot()
    fig.construct(nplots=nplots, ncols=nplots, figsize=figsize)
    fig.modify_spines(ax="all", spines_to_delete=["top", "right"])

    axes = []
    for i in range(nplots):
        axes.append(fig.AxesDict[0][i])

    return fig, axes


def _plot_clone_counts(clone_count_df):

    if type(clone_count_df) != list:
        clone_count_df = [clone_count_df]

    fig, axes = _simple_scatter(figsize=1, nplots=1)
    colors = ["dimgrey", "darkred"]
    labels = ["All clones", "Filtered"]
    for n, _df in enumerate(clone_count_df):
        axes[0].scatter(_df.index, _df["count"], c=colors[n], label=labels[n])
        axes[0].set_xlabel("Clones")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Counted clones")
    axes[0].legend(edgecolor="white")


def _count_clones_adata(adata, clone_id_key):

    clone_count_df = adata.obs.dropna().groupby([clone_id_key]).count()
    sort_by = clone_count_df.columns[-1]
    clone_count_df = clone_count_df.sort_values(sort_by, ascending=False).reset_index()[
        [clone_id_key, sort_by]
    ]
    clone_count_df.columns = ["clone", "count"]

    return clone_count_df


def _count_clones(
    adata1, adata2=False, clone_id_key="clone_idx", plot=False
):

    """Count the number of cells for each clonal barcode"""

    clone_count_df = _count_clones_adata(adata1, clone_id_key)
    if adata2:
        df2 = _count_clones_adata(adata2, clone_id_key)
        clone_count_df = [clone_count_df, df2]

    if plot:
        _plot_clone_counts(clone_count_df)

    return clone_count_df