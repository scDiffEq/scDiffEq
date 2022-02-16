def _get_initial_conditions(distrubution_function, **kwargs):

    """
    Gets initial conidtions for an n-dimensional simulation.
    Executed in the GenericSimulate class

    Parameters:
    -----------
    distrubution_function
        type: function

    **kwargs:
    ---------
    Functions specific to the destribution function.

    Returns:
    --------
    initial_conditions
    """

    initial_conditions = distrubution_function(**kwargs)

    return initial_conditions
