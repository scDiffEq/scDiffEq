
import pandas as pd

def _rank_clusters_by_mean_pseudotime(
    adata,
    time_key="Time point",
    pseudotime_key="cytotrace",
    annotation_key="Annotation",
    kmeans_key=False,
):

    rows = []

    obs_df = adata.obs.reset_index()

    if not kmeans_key:
        kmeans_key = adata.uns["kmeans_key"]

    for group, group_df in obs_df.groupby(kmeans_key):
        m = group_df[pseudotime_key].mean()
        t = (
            group_df.groupby(annotation_key)
            .count()
            .sort_values(time_key, ascending=False)
            .reset_index()
            .head(3)[[annotation_key, time_key]]
        )
        rows.append([group, m, t.iloc[0][annotation_key]])

    df = (
        pd.DataFrame(rows, columns=["cluster", "t", "annot"])
        .sort_values("t", ascending=False)
        .reset_index(drop=True)
    )
    return df