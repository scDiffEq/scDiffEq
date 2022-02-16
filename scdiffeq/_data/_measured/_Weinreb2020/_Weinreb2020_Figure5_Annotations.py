
import numpy as np
import os
import pandas as pd
import pyrequisites as pyrex


BasenameDict = {}
BasenameDict['Time point'] = "timepoints.npy"
BasenameDict['neu_mo_mask'] = "neutrophil_monocyte_trajectory_mask.npy"
BasenameDict['smoothed_groundtruth_from_heldout'] = "smoothed_groundtruth_from_heldout.npy"
BasenameDict['PBA_predictions'] = "PBA_predictions.npy"
BasenameDict['FateID_predictions'] = "FateID_predictions.npy"
BasenameDict['WOT_predictions'] = "WOT_predictions.npy"
BasenameDict['early_cells'] = "early_cells.npy"
BasenameDict['heldout_mask'] = "heldout_mask.npy"
BasenameDict['clonal_fate_matrix'] = "clonal_fate_matrix.npy"


def _calculate_percent_neutrophil_monocyte(
    adata,
):

    """

    Parameters:
    -----------
    clonal_fate_matrix
    
    neu_mo_mask
    
    early
    
    Returns:
    --------
    neu_vs_mo_percent
    has_fate_mask
    d2_neu_mon
    early_neu_mon
    
    Notes:
    ------
    (1) Start by defining and array corresponding to the clonal output of single
        cells represented as %neu vs. %mon. 

    (2) Not all cells have a detected clonal output so we will generate a mask

    (3) Restrict day 2 cells those in the neu/mon - accomplished via mask from (2)
    
    (4) mask that corresponds to the interesection of day 2 and neu/mo cells
    
    (5) mask that corresponds to early cells that are on the mo/nue trajectory
    neu_vs_mo_percent.shape == (1429,)
    has_fate_mask.shape == (28249,)
    early_neu_mon.shape == (28249,)
    """

    clonal_fate_matrix = adata.uns["clonal_fate_matrix"].values
    nonzero_fates = clonal_fate_matrix[:, 5:7].sum(1) > 0
    d2_neu_mo = adata.obs["neu_mo_mask"].values[(adata.obs["Time point"] == 2).values]
    has_fate_mask = np.all([nonzero_fates, d2_neu_mo], axis=0)

    neu_vs_mo_percent = clonal_fate_matrix[has_fate_mask, 5] / clonal_fate_matrix[
        has_fate_mask, 5:7
    ].sum(1)

    early = adata.obs["early_cells"][adata.obs["early_cells"] >= 0]
    early_neu_mo = np.all([early, d2_neu_mo], axis=0)

    d2_idx = adata.obs.loc[adata.obs["Time point"] == 2].index.astype(int)

    for values, column in zip(
        [has_fate_mask, early_neu_mo], ["has_fate_mask", "early_neu_mo"]
    ):
        _tmp_vector = np.full(len(adata), -1)
        _tmp_vector[d2_idx] = values
        adata.obs[column] = _tmp_vector

    has_fate_mask_idx = adata.obs.loc[adata.obs["has_fate_mask"] == 1].index.astype(int)
    _tmp_vector = np.full(len(adata), -1)
    _tmp_vector[has_fate_mask_idx] = neu_vs_mo_percent
    adata.obs["neu_vs_mo_percent"] = _tmp_vector
    

def _adata_obs_loc(adata, column, criteria):
    
    """only considers == criteria, not yet .isin() or other similar examples."""
    
    cols = pyrex.to_list(column)
    crit = pyrex.to_list(criteria)
    
    assert len(cols) == len(crit), print("passed columns and criteria must be of equal length!")
    
    subset_obs = adata.obs
    for i, j in zip(cols, crit):
        subset_obs = subset_obs.loc[subset_obs[i] == j]
    
    return subset_obs

def _annotate_adata_clonal_fate_matrix(adata, clonal_fate_matrix, clonal_fates):
    
    ClonalFateDict = {}
    for n, fate in enumerate(clonal_fates):
        ClonalFateDict[fate] = clonal_fate_matrix[:, n]
        
    adata.uns['clonal_fate_matrix'] = pd.DataFrame.from_dict(ClonalFateDict)
    adata.uns['clonal_fates'] = clonal_fates
    
class _Weinreb2020_Figure5_Annotations:
    
    def __init__(self, directory_path=__file__):

        """Load annotations used in Figure 5 of Weinreb, et al. *Science*. 2020.



        Notes:
        ------

        Predictions:
        (1) Load the predictions from each algorithm. Each set of predictions is a
        vector of length 20157. The length is determined by the intersection of
        neu/mo trajectory cells and day 2 cells

        SPRING plot coordinates:
        (1) Load the dimensionally reduced x,y coordinates for all cells.

        Timepoints:
        (1) Load an array with the time point for each cell (day 2, day 4, or day 6)

        Masks:
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

        Cell fate matrix:
        (1) Load a matrix of clonal fates. Each row corresponds to a day 2 cell, each
            column corresponds to a fate. The value in each entry is the number of day
            4/6 sisters of the day 2 in the particular fate. The columns correspond
            to: Er, Mk, Ma, Ba, Eos, Neu, Mo, MigDC, pDC, Ly
        """

        if directory_path == __file__:
            directory_path = os.path.join(os.path.dirname(directory_path), "_Weinreb2020_Figure5_files")

            self._BasenameDict = BasenameDict
            for key, value in self._BasenameDict.items():
                self.__setattr__(key, np.load(os.path.join(directory_path, value)))

        self.clonal_fates = np.array(["Er", "Mk", "Ma", "Ba", "Eos", "Neu", "Mo", "MigDC", "pDC", "Ly"])
        
def _annotate_adata_with_Weinreb2020_Fig5_predictions(adata):
    
    """"""
    
    Fig5_Annotations = _Weinreb2020_Figure5_Annotations()
    
    pass_list = ['clonal_fates', 'clonal_fate_matrix', '_BasenameDict']
    
    for attr_i in pyrex.class_attributes(Fig5_Annotations):
        attribute = Fig5_Annotations.__getattribute__(attr_i)
        if len(attribute) == adata.shape[0]:
            adata.obs[attr_i] = attribute
            pass_list.append(attr_i)
            
    for attr_i in pyrex.class_attributes(Fig5_Annotations):
        
        if attr_i in pass_list:
            continue
        else:
            
            attribute = Fig5_Annotations.__getattribute__(attr_i)
            n_cells = attribute.shape[0]

            vector = np.full(len(adata), -1, dtype=float)
                
            if n_cells == 28249:
                idx = _adata_obs_loc(adata, "Time point", 2).index.astype(int)    
            elif n_cells == 20157:
                adata.uns['pred_idx'] = idx = _adata_obs_loc(adata, ["Time point", "neu_mo_mask"], [2, True]).index.astype(int)
            else:
                continue
                
            vector[idx] = attribute
            adata.obs[attr_i] = vector
            
    _annotate_adata_clonal_fate_matrix(adata, 
                             Fig5_Annotations.clonal_fate_matrix, 
                             Fig5_Annotations.clonal_fates)
    
    _calculate_percent_neutrophil_monocyte(adata)

    return adata