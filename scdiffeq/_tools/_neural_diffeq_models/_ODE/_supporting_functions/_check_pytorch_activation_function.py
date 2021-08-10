
# local imports #
# ------------- #
from ._enumerate_activation_functions import _enumerate_activation_functions

def _check_pytorch_activation_function(activation_function):

    """
    Asserts that activation function is valid within torch.nn library.

    Parameters:
    -----------
    activation_function

    Returns:
    --------
    None, potential assertion
    """

    assert activation_function.__class__.__name__ in _enumerate_activation_functions(
        return_funcs=True, silent=True
    ), _enumerate_activation_functions(return_funcs=False, silent=False)