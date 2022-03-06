
import adata_sdk
import licorice
import numpy as np

def _create_cell_mask(adata, columns, criteria):

    """"""
    return adata_sdk.obs_loc(adata, columns, criteria).index.astype(int)


def _parse_args_to_mask_dict(MaskDict, key):

    """"""

    if not key in MaskDict.keys():
        print(
            "Plotting interrupted! {} not found in available masks.\nPlease choose from:".format(
                licorice.font_format(str(key), ["BOLD"]),
            )
        )
        for _key in list(MaskDict.keys()):
            print("\t{}".format(licorice.font_format(str(_key), ["BOLD"])))
        return
    else:
        return MaskDict[key]["column"], MaskDict[key]["criteria"]


def _parse_args_to_method_predictions(adata, key, include_ground_truth, prefix="prediction"):

    """"""

    _df = adata.obs
    _cols = adata.obs.filter(regex=prefix).columns.tolist()
    
    if include_ground_truth:
        selection = ["neu_vs_mo_percent", "smoothed_groundtruth_from_heldout"] + _cols
    else:
        selection = ["smoothed_groundtruth_from_heldout"] + _cols

    if not key in selection:
        print(
            "Plotting interrupted! {} not found in available predictions.\nPlease choose from:".format(
                licorice.font_format(key, ["RED", "BOLD"]),
            )
        )
        for _key in list(selection):
            print("\t{}".format(licorice.font_format(_key, ["BOLD"])))
        return
    else:
        return _df[key][_df[key] >= 0]


def _make_overlapping_indices(arr1, arr2):
    """Return overlapping indices / arrays."""
    return arr1[[val in arr2 for val in arr1]]


def _select_cell_predictions_with_mask(cell_idx, cell_preds, mask):
    cell_indices = _make_overlapping_indices(cell_idx, mask)
    return cell_preds[np.where(cell_idx.isin(cell_indices))[0]]


def _return_cells_in_method_predictions(adata, method, include_ground_truth, prefix="prediction"):

    method_predictions = _parse_args_to_method_predictions(adata, method, include_ground_truth, prefix)
    cell_idx, cell_preds = (
        method_predictions.index.astype(int),
        method_predictions.values,
    )
    return cell_idx, cell_preds


def _return_method_predictions(adata, method, mask=False, include_ground_truth=False, prefix="prediction"):
    cell_idx, cell_preds = _return_cells_in_method_predictions(adata, method, include_ground_truth, prefix)
    if not type(mask) == bool:
        return _select_cell_predictions_with_mask(cell_idx, cell_preds, mask)
    else:
        return cell_preds