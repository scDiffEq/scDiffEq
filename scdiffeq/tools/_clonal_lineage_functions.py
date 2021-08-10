def _extract_lineages_from_clone_matrix(clone_matrix):

    """
    Matrix is of shape: cell x lineage

    Function loops through each cell. If a cell has a 1 at any lineage,
    the position of that 1 (i.e., which lineage) is appended to a list.
    If there are no non-zero values for a cell, a value of "False" is
    recorded to the list.

    This function is encapsulated by the function `assign_clonal_lineages`
    that assigns lineages to an AnnData object.

    Parameters:
    -----------
    clone_matrix
        cell x lineage matrix

    Returns:
    --------
    lineage_assignments
        type: list

    """

    lineage_assignments = []

    for cell, lineage in enumerate(clone_matrix):

        if len(np.where(lineage == 1)[0]) > 0:
            lineage_assignments.append(np.where(lineage == 1)[0][0])
        else:
            lineage_assignments.append(False)

    return lineage_assignments


def _assign_clonal_lineages(adata):

    """
    Assign clonal lineages to cells in AnnData.

    Parameters:
    -----------
    adata

    Returns:
    --------
    None
        AnnData object is modified in place.

    """

    adata.obs["clonal_lineage"] = _extract_lineages_from_clone_matrix(
        adata.obsm["X_clone"].toarray()
    )