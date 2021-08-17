
import numpy as np


def _calculate_global_quasi_potential(self, calculation="probability_density"):

    """
    Calculates the quasi-potential.

    Parameters:
    -----------
    calculation
        default: "probability_density"
        type: str

    Returns:
    --------

    Notes:
    ------
    """

    density = self.DensityDict["density"] / self.DensityDict["density"].max()

    if calculation == "probability_density":

        qp = -np.log(density)
        self.quasi_potential = qp_norm = qp / qp.max()