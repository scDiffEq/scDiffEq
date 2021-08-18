import numpy as np
from .supporting_functions.gaussian_distribution import gaussian_distribution


def generate_initial_conditions(number_trajectories, mean_values_vector, variance=0.15):

    """"""

    initial_conditions = np.array([])

    for i, feature_value in enumerate(mean_values_vector):

        single_feature_distribution_all_trajectories = abs(
            gaussian_distribution(number_trajectories, feature_value, variance)
        )
        initial_conditions = np.append(
            initial_conditions, single_feature_distribution_all_trajectories
        )

    initial_conditions = initial_conditions.reshape(i + 1, number_trajectories).T

    return initial_conditions
