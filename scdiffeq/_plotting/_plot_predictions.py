from ..utilities._subsetting_functions import _group_adata_subset
from ._plotting_presets import _single_fig_presets as presets
import matplotlib.pyplot as plt

def _plot_predictions(adata, subset, savename):

    """"""
    
    if subset == "test":
        subset_data = _group_adata_subset(adata, "test", time_name="time")
    elif subset == "training":
        subset_data = _group_adata_subset(adata, "training", time_name="time")
    elif subset == "validation":
        subset_data = _group_adata_subset(adata, "validation", time_name="time")
    else:
        print("A valid subset was not provided.")
    
    try:

        reshaped_pca = adata.uns["predictions_pca"].reshape(
            adata.uns["num_predicted_trajectories"],
            int(
                adata.uns["predictions_pca"].shape[0]
                / adata.uns["num_predicted_trajectories"]
            ),
            2,
        )
        
        presets(title="Predicted Trajectories", x_lab="$PC-1$", y_lab="$PC-2$")
        plt.scatter(
            subset_data.data[:, 0],
            subset_data.data[:, 1],
            c="lightgrey",
            alpha=1,
            zorder=1,
            label="Test data, true trajectories",
        )

        plt.scatter(
            adata.uns["predictions_pca"][:, 0],
            adata.uns["predictions_pca"][:, 1],
            s=2,
            zorder=2,
            alpha=1,
            c="mediumpurple",
            label="Predicted Trajectory",
        )

        plt.scatter(
            reshaped_pca[:, 0, 0],
            reshaped_pca[:, 0, 1],
            s=20,
            zorder=2,
            alpha=1,
            c="palevioletred",
            label="y0",
        )
        print("plotting low-dimensional projection of predictions...")


        
    except:
        reshaped_predictions = adata.uns["predictions"].reshape(
            adata.uns["num_predicted_trajectories"],
            int(
                adata.uns["predictions"].shape[0]
                / adata.uns["num_predicted_trajectories"]
            ),
            2,
        )

        presets(title="Predicted Trajectories", x_lab="$X$", y_lab="$Y$")
        plt.scatter( 
            subset_data.data[:, 0],
            subset_data.data[:, 1],
            c="lightgrey",
            alpha=0.75,
            zorder=1,
            label="Test data, true trajectories",
        )

        plt.scatter(
            adata.uns["predictions"][:, 0],
            adata.uns["predictions"][:, 1],
            s=2,
            zorder=2,
            alpha=1,
            c="navy",
            label="Predicted Trajectory",
        )

        plt.scatter(
            reshaped_predictions[:, 0, 0],
            reshaped_predictions[:, 0, 1],
            s=20,
            zorder=2,
            alpha=1,
            c="orange",
            label="y0",
        )
        
        
        
    leg = plt.legend(markerscale=3, edgecolor="w", fontsize=14, handletextpad=None)
    leg.legendHandles[0]._sizes = [30]
    leg.legendHandles[1]._sizes = [30]
    leg.legendHandles[2]._sizes = [30]

    if savename != None:
        plt.savefig(savename + ".png")
    plt.show()