
import matplotlib.pyplot as plt
import vinplots

from ._static_mask_dict import _static_mask_dict
from ._supporting_cell_parsing_functions import _parse_args_to_mask_dict
from ._supporting_cell_parsing_functions import _create_cell_mask
from ._supporting_cell_parsing_functions import _return_method_predictions

def _plot_background_cells(
    adata,
    axes,
    MaskDict,
    background_mask,
):

    background_mask_columns, background_mask_criteria = _parse_args_to_mask_dict(
        MaskDict, background_mask
    )

    selected_background_mask = _create_cell_mask(
        adata, background_mask_columns, background_mask_criteria
    )
    umap_coords = adata.obsm["X_umap"][selected_background_mask]
    u_x, u_y = umap_coords[:, 0], umap_coords[:, 1]
    for ax in axes:
        ax.scatter(
            u_x,
            u_y,
            c="lightgrey",
            s=2,
        )


def _plot_predicted_cells(
    adata, axes, MaskDict, method_predictions_df, predicted_mask, colormap, include_ground_truth=False, prefix="prediction"
):

    predicted_mask_columns, predicted_mask_criteria = _parse_args_to_mask_dict(
        MaskDict, predicted_mask
    )
    selected_predicted_mask = _create_cell_mask(
        adata, predicted_mask_columns, predicted_mask_criteria
    )

    umap_coords = adata.obsm["X_umap"][selected_predicted_mask]
    u_x, u_y = umap_coords[:, 0], umap_coords[:, 1]

    if include_ground_truth:
        plot_cols = [
            "neu_vs_mo_percent", "smoothed_groundtruth_from_heldout"
        ] + method_predictions_df.columns.tolist()
    else:
        plot_cols = ["smoothed_groundtruth_from_heldout"] + method_predictions_df.columns.tolist()
    for n, method in enumerate(plot_cols):
        prediction_values = _return_method_predictions(
            adata, method, selected_predicted_mask, include_ground_truth, prefix,
        )
        ax = axes[n]
        im = ax.scatter(
            u_x, u_y, c=prediction_values, s=15, cmap=colormap, vmin=0.1, vmax=0.9
        )
        if method is "neu_vs_mo_percent":
            title_string = "True N/M Fate Count"
            
        elif method is "smoothed_groundtruth_from_heldout":
            title_string = "Smoothed Ground Truth"
            _im = im
        else:
            title_string = method.split("_")[0]
        ax.set_title(title_string, fontsize=24)

    n_cells_pred = len(selected_predicted_mask)

    return _im, n_cells_pred

def _plot_colorbar(im, ax):

    cbar = plt.colorbar(mappable=im, ax=ax, location="top", ticks=[0, 1])
    cbar.outline.set_visible(False)
    cbar.ax.set_xticklabels(["Mon", "Neu"])
    cbar.set_label(
        "Clonal fate bias",
    )
    
def _build_plot_highlight_population():

    """"""

    fig = vinplots.Plot()
    fig.construct(nplots=1, ncols=1, figsize_width=0.9)
    fig.modify_spines(ax="all", spines_to_delete=["top", "right", "left", "bottom"])
    ax = fig.AxesDict[0][0]
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def _plot_highlight_predicted_cells(
    adata,
    MaskDict,
    background_mask="neu_mo",
    predicted_mask="early_d2_neu_mo",
    highlight_color="#333333",
    save=False,
    dpi=200,
    show=True,
):

    h_fig, h_ax = _build_plot_highlight_population()

    predicted_mask_columns, predicted_mask_criteria = _parse_args_to_mask_dict(
        MaskDict, predicted_mask
    )
    selected_predicted_mask = _create_cell_mask(
        adata, predicted_mask_columns, predicted_mask_criteria
    )

    umap_coords = adata.obsm["X_umap"][selected_predicted_mask]
    u_x, u_y = umap_coords[:, 0], umap_coords[:, 1]
    n_cells_pred = len(selected_predicted_mask)
    
    if background_mask:
        _plot_background_cells(
            adata, axes=[h_ax], MaskDict=_static_mask_dict(), background_mask=background_mask
        )
    h_ax.scatter(u_x, u_y, c=highlight_color, s=2)
    h_ax.set_title("# Cells: {}".format(n_cells_pred), fontsize=12)
    
    if save:
        h_fig.fig.dpi = dpi
        plt.savefig("background_{}.prediction_{}.highlight.{}dpi.png".format(str(background_mask),
                                                                             str(predicted_mask),
                                                                             str(dpi)))
    if not show:
        plt.close()