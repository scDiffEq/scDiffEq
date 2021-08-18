import numpy as np


def gaussian_distribution(number_trajectories, variance=0.15):

    mean = 0
    distribution = np.random.normal(mean, variance, number_trajectories) * 2
    return distribution


def random_distribution(number_trajectories):

    """Generates a random distribution around zero."""

    distribution = np.random.random(number_trajectories) * 2 - 1

    return distribution


def axis_locked_negative(number_trajectories):

    distribution = np.ones(number_trajectories) * -1

    return distribution


def axis_locked_positive(number_trajectories):

    distribution = np.ones(number_trajectories) * 1

    return distribution


def axis_zeroed(number_trajectories):

    distribution = np.ones(number_trajectories) * 0

    return distribution


def get_initial_conditions(keywords, number_trajectories, variance=0.15):

    initial_conditions = np.array([])

    for i in range(len(keywords)):

        available_distributions = {
            "gaussian": gaussian_distribution(number_trajectories, variance=0.15),
            "random": random_distribution(number_trajectories),
            "axis-locked-negative": axis_locked_negative(number_trajectories),
            "axis-locked-positive": axis_locked_positive(number_trajectories),
            "axis-zeroed": axis_zeroed(number_trajectories),
        }

        distribution = available_distributions[keywords[i]]
        initial_conditions = np.append(distribution, initial_conditions)

    initial_conditions = initial_conditions.reshape(number_trajectories, len(keywords))

    return initial_conditions
