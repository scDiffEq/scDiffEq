
import licorice

def _print_verbose_function_documentation(imported_function, package, module, function):

    """
    Prints the documentation of a python function imported using v.ut.import_from_pystring()

    Parameters:
    -----------
    imported_function [ required ]
        python function imported using v.ut.import_from_pystring()
        type: function

    package [ required ]
        default: 'numpy'
        type: str

    module [ required ]
        default: 'random'
        type: str

    function [ required ]
        Name of the function as implemented in numpy.random.<function>
        type: str

    Returns:
    --------
    None

    Notes:
    ------
    """

    preamble = "\nDocumentation for the chosen function:"
    function_name = ".".join([package, module, function])

    print(
        "{} {}".format(
            licorice.font_format(preamble, ["BOLD",],),
            licorice.font_format(function_name, ["BOLD", "RED"]),
        )
    )
    print(imported_function.__doc__)
