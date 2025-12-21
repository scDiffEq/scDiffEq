
import pandas as pd
import numpy as np

from ._mesh_grid import MeshGrid
from ...core import utils
import ABCParse

class Surface3D(ABCParse.ABCParse):
    """
    Mesh grid surface.
    
    Uses a passed mesh grid and pre-fit neighbor model to map
    the points of a mesh grid to the corresponding y-val.
    """

    def __init__(self, gridpoints: int = 250):

        self.__parse__(locals(), public=[None])

    def _build_mesh_grid(self):
        self._mesh_grid = MeshGrid(
            adata=self.adata, gridpoints=self._gridpoints
        )
        self._x, self._y = self._mesh_grid()
        return self._mesh_grid

    @property
    def X(self):
        return self._x.reshape(-1, 1)

    @property
    def Y(self):
        return self._y.reshape(-1, 1)

    @property
    def mesh_grid_coordinates(self):
        return np.concatenate([self.X, self.Y], axis=1)
    
    def _to_frame(self, z_pred):
        self._surface_df = pd.DataFrame(
            data=z_pred, index=self._mesh_grid._X_GRID, columns=self._mesh_grid._Y_GRID
        )
        return self._surface_df
        

    def predict_from_model(self):
        return self.neighbor_model.predict(self.mesh_grid_coordinates).reshape(
            self._gridpoints, self._gridpoints
        )

    def __call__(self, adata, neighbor_model):
        """ """
        
        self.__update__(locals())
        self._build_mesh_grid()
        
        return self._to_frame(self.predict_from_model())
