# package imports #
# --------------- #
import numpy as np


def _four_attractor_2d_state_equation(state, time):

    """
    State function for simulating a four-attractor dynamical system in 2-D.

    Parameters:
    -----------
    time
        type: numpy.ndarray

    states
        type: numpy.ndarray

    Returns:
    --------
    state_vector
        array of state variables.
        type: numpy.ndarray
    """

    x, y = state[0], state[1]

    x_dot = -1 + (9 * x) - (2 * (x ** 3)) + (9 * y) - (2 * (y ** 3))
    y_dot = 1 - (11 * x) + (2 * (x ** 3)) + (11 * y) - (2 * (y ** 3))

    state_vector = np.array([x_dot, y_dot])

    return state_vector
