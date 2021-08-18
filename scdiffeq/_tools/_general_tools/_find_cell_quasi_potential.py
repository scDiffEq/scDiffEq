import numpy as np


def _find_cell_quasi_potential(self):

    """Get QP for each cell based on nearest neighbor to grid."""

    x_mesh, y_mesh = self.DensityDict["x_mesh"], self.DensityDict["y_mesh"]
    xy = np.array([self.DensityDict["x"], self.DensityDict["y"]]).T

    cell_qp = []
    for i in xy:
        hx = abs(i[0] - x_mesh)
        hy = abs(i[1] - y_mesh)
        h_xx, h_xy = np.unique(np.argmin(hx, axis=0)), np.unique(np.argmin(hx, axis=1))
        h_yx, h_yy = np.unique(np.argmin(hy, axis=0)), np.unique(np.argmin(hy, axis=1))
        cell_qp.append(self.quasi_potential[h_xx, h_yy][0])

    self.cell_quasi_potential = np.array(cell_qp)
