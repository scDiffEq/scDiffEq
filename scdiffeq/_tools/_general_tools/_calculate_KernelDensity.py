from sklearn.neighbors import KernelDensity
import numpy as np
import glob, os, torch
import vintools as v


def _calculate_kernel_density_2d(data, nbins=200, bandwidth=1, widen=0):

    xbins = ybins = complex(0, nbins)

    x_, y_ = data[:, 0], data[:, 1]

    mesh_x, mesh_y = np.mgrid[
        x_.min() - widen : x_.max() + widen : xbins,
        y_.min() - widen : y_.max() + widen : ybins,
    ]

    xy_sample = np.vstack([mesh_y.ravel(), mesh_x.ravel()]).T
    xy_train = np.vstack([y_, x_]).T

    kde_skl = KernelDensity(bandwidth=bandwidth)
    kde_skl.fit(xy_train)

    z = np.exp(kde_skl.score_samples(xy_sample))
    density = np.reshape(z, mesh_x.shape)

    return mesh_x, mesh_y, density


def _read_uns_tensors(self, return_dict=False, silent=False):

    TensorDict = {}

    uns_file_contents = glob.glob(self._uns_path + "/*.pt")
    for file in uns_file_contents:
        filename = os.path.basename(file.strip(".pt"))
        TensorDict[filename] = torch.load(file)

    if not silent:
        print(TensorDict.keys())

    self.TensorDict = TensorDict

    return TensorDict


def _xy_3d_to_2d(array, silent=False):

    """"""

    axis_0 = array.shape[0]
    axis_1 = array.shape[1]
    axis_2 = array.shape[2]

    reshaped_array = array.reshape(axis_0 * axis_1, axis_2)

    if not silent:
        str1 = v.ut.format_pystring("Reshaped from: ", ["BOLD"])
        str2 = v.ut.format_pystring(" to: ", ["BOLD"])
        print("{}{}{}{}".format(str1, array.shape, str2, reshaped_array.shape))

    return reshaped_array


from ..._plotting._plot_kernel_density import _KernelDensity_plot_presets


def _calculate_KernelDensity(
    self, clear_DensityDict=False, figure_legend_loc=4, plot=True
):

    """
    Calculate and plot the 2D kernel density estimate (KDE) from predicted data.
    """

    if clear_DensityDict:
        del self.DensityDict

    try:
        self.DensityDict
        print("DensityDict identified... plotting.")
    except:
        self.DensityDict = {}

        try:
            y_pred = _xy_3d_to_2d(
                _read_uns_tensors(self, return_dict=True)["test_y_predicted"]
            ).numpy()
        except:
            y_pred = _xy_3d_to_2d(self.adata.uns["test_y_predicted"]).numpy()
        self.DensityDict["x"], self.DensityDict["y"] = y_pred[:, 0], y_pred[:, 1]
        (
            self.DensityDict["x_mesh"],
            self.DensityDict["y_mesh"],
            self.DensityDict["density"],
        ) = _calculate_kernel_density_2d(y_pred)

    if plot:
        _KernelDensity_plot_presets(
            self.DensityDict, figure_legend_loc=figure_legend_loc,
        )
