
import os
import numpy as np

def _get_neu_mon_percent(data):
    """
    (1) Start by defining and array corresponding to the clonal output of single
        cells represented as %neu vs. %mon. 

    (2) Not all cells have a detected clonal output so we will generate a mask

    (3) Restrict day 2 cells those in the neu/mon - accomplished via mask from (2)
    
    (4) mask that corresponds to the interesection of day 2 and neu/mo cells
    
    (5) mask that corresponds to early cells that are on the mo/nue trajectory
    """
    
    neu_mo_mask = data.masks['neu_mo_mask']
    early = data.masks['early_cells']
    clonal_fate_matrix = data.clonal_fate_matrix
    
    has_fate_mask = np.all([clonal_fate_matrix[:,5:7].sum(1) > 0, neu_mo_mask[data.timepoints==2]],axis=0)
    neu_vs_mo_percent = clonal_fate_matrix[has_fate_mask,5] / clonal_fate_matrix[has_fate_mask,5:7].sum(1)    
    d2_neu_mon = np.all([neu_mo_mask, data.timepoints == 2], axis=0)
    
    early_neu_mon = np.all([early, neu_mo_mask[data.timepoints==2]],axis=0)
    
    return neu_vs_mo_percent, has_fate_mask, d2_neu_mon, early_neu_mon

def _load_cell_masks(directory):

    """

    (1) Load the mask that corresponds to just the neutrophil/mono trajectory

    (2) In order to recirulate the figure, we also need to load a mask that
        corresponds to early (Cd34+ cells). The mask has length 28249, which
        is equal to the total number of day 2 cells

    (3, 4) In the figure, we compare the accuracy of fate prediction to a clonal benchmark determined by
        the values of heldout clonal data. Here we load the heldout mask and the smoothed 'groundtruth'
        Both arrays have length 20157, which is the intersection of neu/mo cells and day 2 cells.

    (5) Finally, to exactly reproduce the figure, we need to load a mask which excludes outlier cells in
        the SPRING plot, and was used to make the visualization more clear. This array also has length
        20157 and has 2% positive values (i.e., 2% of cells are excluded)
    """

    CellMaskDict = {}
    CellMaskDict["neu_mo_mask"] = np.load(
        os.path.join(directory, "neutrophil_monocyte_trajectory_mask.npy")
    )
    CellMaskDict["early_cells"] = np.load(os.path.join(directory, "early_cells.npy"))
    CellMaskDict["heldout_mask"] = np.load(os.path.join(directory, "heldout_mask.npy"))
    CellMaskDict["smoothed_groundtruth"] = np.load(
        os.path.join(directory, "smoothed_groundtruth_from_heldout.npy")
    )
    CellMaskDict["outlier_mask"] = np.load(
        os.path.join(directory, "outliers_in_SPRING_plot.npy")
    )

    return CellMaskDict


def _load_clonal_fate_matrix(directory):

    """

    Notes:
    ------
    (1) Load a matrix of clonal fates. Each row corresponds to a day 2 cell, each
        column corresponds to a fate. The value in each entry is the number of day
        4/6 sisters of the day 2 in the particular fate. The columns correspond
        to: Er, Mk, Ma, Ba, Eos, Neu, Mo, MigDC, pDC, Ly

    """

    clonal_fate_matrix = np.load(os.path.join(directory, "clonal_fate_matrix.npy"))

    return clonal_fate_matrix


def _load_timepoints(directory):

    """
    Load an array with the time point for each cell (day 2, day 4, or day 6)
    """
    return np.load(os.path.join(directory, "timepoints.npy"))


def _load_spring_coordinates(directory):

    """Load the dimensionally reduced x,y coordinates for all cells."""

    x, y = np.load(os.path.join(directory, "coordinates_x.npy")), np.load(
        os.path.join(directory, "coordinates_y.npy")
    )
    X_ = np.stack([x, y]).T

    return X_


def _load_external_predictions(directory):

    """
    Load the predictions from each algorithm. Each set of predictions is a
    vector of length 20157. The length is determined by the intersection of
    neu/mo trajectory cells and day 2 cells
    """

    filepaths = {
        "smoothed": "smoothed_groundtruth_from_heldout.npy",
        "PBA": "PBA_predictions.npy",
        "FateID": "FateID_predictions.npy",
        "WOT": "WOT_predictions.npy",
    }
    PredictionDict = {}

    for algorithm, fpath in filepaths.items():
        PredictionDict[algorithm] = np.load(os.path.join(directory, fpath))

    return PredictionDict


class _Weinreb2020Predictions:
    def __init__(self, data_directory):

        self.dir = data_directory

    def load(self):

        self.masks = _load_cell_masks(self.dir)
        self.clonal_fate_matrix = _load_clonal_fate_matrix(self.dir)
        self.fates = np.array(["Er", "Mk", "Ma", "Ba", "Eos", "Neu", "Mo", "MigDC", "pDC", "Ly"])
        self.timepoints = _load_timepoints(self.dir)
        self.coords = _load_spring_coordinates(self.dir)
        self.predictions = _load_external_predictions(self.dir)
        
    def get_neu_mon_percent(self):
        
        self.neu_vs_mo_percent, self.has_fate_mask, self.d2_neu_mo_mask, self.early_neu_mon = _get_neu_mon_percent(self)