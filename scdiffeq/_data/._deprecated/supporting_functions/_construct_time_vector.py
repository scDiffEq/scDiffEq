def zero_normalize_noisy_time_vector(noisy_time_vector):

    """
    if the first element of the noisy time vector happpens to be less than 0, this function will set replace the negative initial time value with a zero-value. 
    
    Parameters:
    -----------
    noisy_time_vector
        the first value may or may not be greater than or equal to zero.
    
    Returns:
    --------
    noisy_time_vector
        the first value greater than or equal to zero.
    
    """

    if noisy_time_vector[0] < 0:
        noisy_time_vector[0] = 0

    return noisy_time_vector


def make_time_noisy(time_vector, time_step_span, noise_amplitude=0.001):

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
    
    
    """

    import numpy as np

    if noise_amplitude != 0:
        noisy_time_vector = time_vector + np.random.normal(
            0, noise_amplitude, time_vector.shape
        )

    zero_ensured_noisy_time_vector = zero_normalize_noisy_time_vector(noisy_time_vector)

    return noisy_time_vector


def get_single_time_vector(time_span=10.0, number_time_samples=1000):

    """
    Given a time-span, this function creates noisy samplings from within this span across the desired number of samples. 
    
    Parameters:
    -----------
    time_span
        the number of time units over which one wishes to simulate. 
        
    number_time_samples
        the number of samples to be taken within the defined time_span
    
    Returns:
    --------
    noisy_time_vector
        a vector of length = `number_time_samples` spanning `time_span` 
    
    """

    import numpy as np

    time_step_span = time_span / number_time_samples
    time_vector = np.arange(0.0, time_span, time_step_span)
    noisy_time_vector = make_time_noisy(time_vector, time_step_span)

    return noisy_time_vector


def construct_time_vector(number_trajectories, time_span, number_time_samples):

    import numpy as np

    time = np.array([])

    for i in range(number_trajectories):
        one_cell_time_vector = get_single_time_vector(time_span, number_time_samples)
        time = np.append(time, one_cell_time_vector)

    return time
