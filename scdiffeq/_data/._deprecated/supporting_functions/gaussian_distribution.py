import numpy as np


def gaussian_distribution(number_trajectories, mean, variance=0.15):

    distribution = np.random.normal(mean, variance, number_trajectories)

    return distribution
