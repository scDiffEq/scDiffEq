import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_pca_by_trajectory(adata, size):

    for i in range(int(adata.uns["number_of_trajectories"])):

        single_trajectory = adata.obs.loc[adata.obs["trajectory"] == i].index.values.astype(
            int
        )
        single_traj_pca = adata.obsm["X_pca"][single_trajectory]

        plt.scatter(single_traj_pca[:, 0], single_traj_pca[:, 1], s=size)

def _plot_principle_component_analysis(adata, size, colorby):
    
    presets("PCA", x_lab="PC-1", y_lab="PC-2", size=(8,6))
    
    if colorby=="time":
        plt.scatter(adata.obsm["X_pca"][:, 0], adata.obsm["X_pca"][:, 1], c=adata.obs.time.values, s=size)
        cb = plt.colorbar()
    elif colorby=="trajectory":
        plot_pca_by_trajectory(adata, size)

def _sklearn_pca(adata, number_components):
    
    """Runs sklearn PCA."""
    
    from sklearn.decomposition import PCA
        
    try:
        high_dimensional_matrix = adata.X.toarray()
    except:
        high_dimensional_matrix = adata.X
    pca = PCA(number_components)
    pcs = pca.fit_transform(high_dimensional_matrix)
    
    adata.uns["pca"] = pca
    
    pc_df = pd.DataFrame(data=pcs)
    pc_df.columns = ["PC_" + str(pc + 1) for pc in pc_df.columns]
    pc_df.index = adata.obs_names
    
    return pc_df

def principle_component_analysis(adata, number_components=2, plot=False, colorby="time", title="PCA", return_df=False, size=15):

    """
    Performs principle component analysis (PCA). Recieves a data matrix and desired component output number.
    
    Parameters
    ----------
    adata (required)
        path to an h5ad object to be loaded as AnnData.
        
    number_components (required)
        number of principle components to be calculated. 
        default: 2
    
    plot (optional)
        default: False
    
    title (optional)
        default: "PCA"
    
    return_df (optional)
        default: False
    """
    
    pc_df = _sklearn_pca(adata, number_components)
    adata.obsm["X_pca"] = np.array(pc_df)
    print(adata)
    
    if plot == True:
        _plot_principle_component_analysis(adata, size, colorby)
    
    if return_df == True:
        return pc_df