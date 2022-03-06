def _count_annotations(adata, annotation_key="Annotation"):

    count_col = adata.obs.columns[-1]
    obs_df = adata.obs.dropna()
    celltype_counts = (
        obs_df.groupby(annotation_key)
        .count()
        .sort_values(count_col, ascending=False)[count_col]
        .to_frame()
    )
    celltype_counts.columns = ["count"]

    return celltype_counts