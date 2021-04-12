
def ensure_array(adata):

    """
    Parameters:
    -----------
    adata
        AnnData object

    Returns:
    --------
    None, modified in place.
    """

    try:
        adata.X = adata.X.toarray()
    except:
        pass