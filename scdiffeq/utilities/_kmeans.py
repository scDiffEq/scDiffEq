from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from ..plotting.plotting_presets import single_fig_presets as fig_presets
from ..plotting.plotting_presets import annotate_scatterplot

def get_kmeans_inertia(X, k_max=11):

    """
    Calculates the inertia of various $k$-means clustering solutions at different values of $k$.
    
    Parameters:
    -----------
    X
        Data
    k_max
        Max number of $k$ to try. 
    """

    

    wcss = []
    for i in range(1, k_max):
        kmeans = KMeans(
            n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0
        )
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    fig_presets(
        title="$k$-means Clustering Elbow Method",
        x_lab="Number of clusters ($k$)",
        y_lab="Inertia (WCSS)",
        size=(6, 4),
    )
    plt.plot(range(1, k_max), wcss, lw="4", c="mediumpurple", alpha=0.5)
    plt.show()
    
def kmeans(X, k, annotations, magnitude=[[1, 1], [1, 1], [1, 1]], direction=[[1, 1], [1, 1], [1, 1]],):

    """
    
    Parameters:
    -----------
    X
        Data
    k
        Number of centroids
    
    annotations
        list of annotations with which to label centroids
    
    
    Returns:
    --------
    None
        Prints a plot.
    """

    from sklearn.cluster import KMeans

    kmeans = KMeans(
        n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=0
    )
    pred_y = kmeans.fit_predict(X)
    fig_presets(title="$k$-means Clustering", x_lab="$X$", y_lab="$Y$", size=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], s=25, c="olivedrab", label="Data")
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=125,
        c="palevioletred",
        label="Centroids",
    )
    legend_ = plt.legend(edgecolor="w", fontsize=14, handletextpad=None)
    legend_.legendHandles[0]._sizes = [60]
    legend_.legendHandles[1]._sizes = [60]

    annotate_scatterplot(
        kmeans.cluster_centers_,
        annotations,
        magnitude,
        direction,
        point_offset=0,
    )

    plt.show()