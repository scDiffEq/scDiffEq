
import licorice

def _check_if_scipy_sparse_mtx(array):

    """
    Checks if input array is 'coo' or 'csr' sparse matrix. Outputs string indicating status.
    
    Parameters:
    -----------
    array
        dtype is ambiguous before running this function. 
    Returns:
    --------
    scipy.sparse type information
        ['csr','coo','not_scipy.sparse']
    """

    try:
        if array.getformat() == "csr":
            return "csr"

        if array.getformat() == "coo":
            return "coo"

    except:
        return "not_scipy.sparse"


def _format_AnnData_mtx_as_numpy_array(adata, silent=False):

    """
    Checks:
        (1) scipy.stats.sparse_matrix format
        (2) numpy.ndarray format
    Then:
        Converts adata.X to numpy.ndarray or reports that the array was not able to be converted. 
    
    Parameters:
    -----------
    adata

    Returns:
    --------
    None
        AnnData is modified in place. adata.X is converted to a numpy.ndarray.
    """

    array_is_not_status_list = ["\nadata.X is not:"]

    # check if adata.X is a scipy sparse matrix of some sort
    scipy_status = _check_if_scipy_sparse_mtx(adata.X)
    if scipy_status in ["coo", "csr"]:
        out_message = "scipy.sparse.{}_matrix".format(scipy_status)
        if not silent:
            print(
                "adata.X was of dtype: {}.".format(
                    licorice.font_format(out_message, ["RED", "BOLD"])
                )
            )
        adata.X = adata.X.toarray()
        if not silent:
            print(
                "adata.X has been converted to dtype: {}.".format(
                    licorice.font_format("numpy.ndarray", ["RED", "BOLD"])
                )
            )
        return None
    else:
        array_is_not_status_list.append("scipy.sparse.csr_matrix")
        array_is_not_status_list.append("scipy.sparse.coo_matrix")

    # check if numpy array
    if type(adata.X).__name__ == "ndarray":
        if not silent:
            out_message = "adata.X"
            print(
                "{} is already of dtype: numpy.ndarray".format(
                    licorice.font_format(out_message, ["RED", "BOLD"])
                )
            )
    else:
        try:
            adata.X = adata.X.toarray()
            out_message = "adata.X"
            if not silent:
                print(
                    "{} converted to dtype: numpy.ndarray".format(
                        licorice.font_format(out_message, ["RED", "BOLD"])
                    )
                )
        except:
            array_is_not_status_list.append("np.ndarray")
            if not silent:
                print("adata.X was unable to be converted to an array.")
            for i, term in enumerate(array_is_not_status_list):
                if i != 0:
                    if not silent:
                        print(
                            "\n\t{}".format(licorice.font_format(term, ["BOLD", "RED"]))
                        )
                else:
                    print(term)
