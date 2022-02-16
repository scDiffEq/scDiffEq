# package imports #
# --------------- #
import numpy as np


def _add_time_noise(time_vector, time_step_span, noise_amplitude=0.001):

    """
    Adds an element of random noise to the length of each time step.

    Parameters:
    -----------
    time_vector
        a vector containing all increasing time stamps

    time_step_span
        the magnitude of each time step

    noise_amplitude
        multiplier of the added noise amplitude
        default = 0.001

    Returns:
    --------
    noise_adjusted_time_vector

    Notes:
    ------
    If the first element of the noisy time vector happpens to be less than 0, this function
    will set replace the negative initial time value with a zero-value.
    """

    if noise_amplitude != 0:
        noise_adjusted_time_vector = time_vector + np.random.normal(
            0, noise_amplitude, time_vector.shape
        )
    else:
        noise_adjusted_time_vector = time_vector

    noise_adjusted_time_vector[0] = 0

    return noise_adjusted_time_vector


def _create_time_vector(time_span=10.0, n_samples=1000, noise_amplitude=0):

    """
    Given a time-span, this function creates noisy samplings from within this span across the desired number of samples.

    Parameters:
    -----------
    time_span
        the number of time units over which one wishes to simulate.

    n_samples
        the number of samples to be taken within the defined time_span

    Returns:
    --------
    noisy_time_vector
        a vector of length = `number_time_samples` spanning `time_span`

    """

    time_step_span = time_span / n_samples
    time_vector = np.arange(0.0, time_span, time_step_span)
    noise_adjusted_time_vector = _add_time_noise(
        time_vector, time_step_span, noise_amplitude=noise_amplitude
    )

    return noise_adjusted_time_vector
