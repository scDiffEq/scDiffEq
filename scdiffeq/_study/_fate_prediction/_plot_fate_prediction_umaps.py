
import matplotlib.pyplot as plt
import vinplots

from ._static_mask_dict import _static_mask_dict
from ._build_plots import _build_plots
from ._plotting_functions import _plot_background_cells
from ._plotting_functions import _plot_predicted_cells
from ._plotting_functions import _plot_colorbar
from ._plotting_functions import _plot_highlight_predicted_cells

class _FateMappedPredictions:
    def __init__(self, adata, pred_regex="predictions", plot=True):

        """"""

        self._adata = adata
        self._cmap = vinplots.palettes.BlueOrange()
        self._predictions = adata.obs.filter(regex=pred_regex)
        self._n_preds = self._predictions.shape[1]
        self._MaskDict = _static_mask_dict()

    def plot(
        self,
        background_mask="neu_mo",
        prediction_mask="d2_neu_mo_no",
        include_ground_truth=False,
        prefix="prediction",
        save=False,
        dpi=200
    ):
        
        if include_ground_truth:
            add_n = 2
        else:
            add_n = 1
            
        self._fig, self._axes, self._bottom_row = _build_plots(self._n_preds + (add_n-1))
        
        if background_mask:
            _plot_background_cells(
                self._adata,
                self._axes[: int(self._n_preds + add_n)],
                self._MaskDict,
                background_mask,
            )
        self._im, self._n_cells_pred = _plot_predicted_cells(
            self._adata,
            self._axes[: int(self._n_preds + add_n)],
            self._MaskDict,
            self._predictions,
            prediction_mask,
            self._cmap,
            include_ground_truth,
            prefix,
        )
        _plot_colorbar(im=self._im, ax=self._fig.AxesDict[self._bottom_row][0])
        

    def highlight_subpopulation(
        self, background_mask="neu_mo", mask="early_d2_neu_mo", highlight_color="#333333", save=False, dpi=200, show=True,
    ):
        _plot_highlight_predicted_cells(
            self._adata,
            self._MaskDict,
            background_mask=background_mask,
            predicted_mask=mask,
            highlight_color=highlight_color,
            save=save,
            dpi=dpi,
            show=show,
        )
        
def _plot_predicted_fates(adata,
                          background_mask="neu_mo",
                          prediction_mask="d2_neu_mo",
                          highlight_color="#333333",
                          include_ground_truth=False,
                          prefix="prediction",
                          save=False,
                          dpi=200,
                          show=True,
                         ):
    
    FateMapped = _FateMappedPredictions(adata)
    FateMapped.plot(
        background_mask=background_mask,
        prediction_mask=prediction_mask,
        include_ground_truth=include_ground_truth,
        prefix=prefix,
        save=save,
        dpi=dpi,
    )
                        
    FateMapped.highlight_subpopulation(background_mask=background_mask, 
                                       mask=prediction_mask,
                                       highlight_color=highlight_color,
                                       save=save,
                                       dpi=dpi,
                                       show=show,
                                      )
    if save:
        FateMapped._fig.fig.dpi = dpi
        plt.savefig("background_{}.prediction_{}.{}dpi.png".format(str(background_mask), str(prediction_mask), str(dpi)))
        if not show:
            plt.close()