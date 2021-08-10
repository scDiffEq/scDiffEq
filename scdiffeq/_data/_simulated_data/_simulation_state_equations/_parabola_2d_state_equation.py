
# package imports #
# --------------- #
import numpy as np

def _parabola_2d_state_equation(state, time):

    """
    State function for simulating a single-attractor parabola in 2-D.

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
    
    Notes:
    ------
    It's a bit confusing, but this function is generally used within scipy.integrate.odeint(func, y0, t)
    where func is this function. 
    """

    x, y = state[0], state[1]

    x_dot = -x + y ** 2
    y_dot = -y - x * y

    state_vector = np.array([x_dot, y_dot])

    return state_vector