
__module_name__ = "_determine_input_data_dim.py"
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])



def _determine_input_data(adata, use_key="X", layer=None):

    if layer:
        return adata.layers[layer]

    elif use_key == "X":
        return adata.X.copy()

    elif use_key in adata.obsm_keys():
        return adata.obsm[use_key]