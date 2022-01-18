
import os

def _Weinreb2020_PathDict(path=""):

    """"""

    PathDict = {}

    PathDict["X"] = "counts_matrix_in_vitro.npz"
    PathDict["clonal"] = "clone_annotation_in_vitro.npz"
    PathDict["obs"] = "cell_metadata_in_vitro.txt"
    PathDict["var"] = "gene_names_in_vitro.txt"
    PathDict["X_spring"] = "coordinates_in_vitro.txt"

    for key, filename in PathDict.items():
        path_filename = os.path.join(path, filename)
        PathDict[key] = path_filename

    return PathDict