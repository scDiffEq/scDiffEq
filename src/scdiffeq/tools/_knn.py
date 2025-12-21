# -- Re-export kNN from cell-neighbors: ----------------------------------------
# This module previously contained a custom kNN implementation using annoy.
# It now re-exports the kNN class from cell-neighbors which uses voyager.

from cell_neighbors import kNN

__all__ = ["kNN"]
