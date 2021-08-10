
# package imports #
# --------------- #
import vintools as v

# local imports #
# ------------- #
from ._print_verbose_function_documentation import _print_verbose_function_documentation

def _get_numpy_distribution_function(
    function, module="random", package="numpy", print_verbose_documentation=False
):

    """
    Parameters:
    -----------
    function [ required ]
        Name of the function as implemented in numpy.random.<function>
        type: str

    module
        default: 'random'
        type: str

    package
        default: 'numpy'
        type: str

    Returns:
    --------
    np_distribution_func
        type: function

    Notes:
    ------
    Common options for 'function' are:
        (1) 'normal'
        (2) 'random'
    """

    np_distribution_func = v.ut.import_from_string(
        package=package, module=module, function=function
    )

    if print_verbose_documentation:
        _print_verbose_function_documentation(
            imported_function=np_distribution_func,
            package=package,
            module=module,
            function=function,
        )

    return np_distribution_func