from importlib import import_module


def _dynamical_import_of_function_from_string(
    package, module, function, function_parameters=None
):

    """
    Import a specific function from an installed package, dynamically using a string.
    Parmeters:
    ----------
    package
        Name of the package
        type: str
    module
        Name of the module in the package
        type: str
    function
        Name of the function in the module within the package
        type: str
    
    function_parameters
        Default: None
    Returns:
    --------
    function <package.module.function>
    """

    package_module = ".".join([package, module])

    module = import_module(name=package_module)

    # some functions need parameters; others require not having parameters
    try:
        function = getattr(module, function)
    except:
        function = getattr(module, function)(function_parameters)

    return function