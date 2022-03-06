

def _static_mask_dict():

    """"""

    MaskDict = {}
    MaskDict["neu_mo"] = {}
    MaskDict["neu_mo_no_outliers"] = {}
    MaskDict["early_d2_neu_mo"] = {}
    MaskDict["d2_neu_mo_no_outliers"] = {}
    MaskDict["d2_neu_mo"] = {}
    MaskDict["fate_mask_d2"] = {}
    MaskDict["filtered_fate_heldout_0"] = {}
    MaskDict["filtered_fate_heldout_1"] = {}
    MaskDict["filtered_fate_early_0"] = {}
    MaskDict["filtered_fate_early_1"] = {}
    MaskDict["filtered_fate_heldout_0_early_0"] = {}
    MaskDict["filtered_fate_heldout_1_early_0"] = {}
    MaskDict["filtered_fate_heldout_0_early_1"] = {}
    MaskDict["filtered_fate_heldout_1_early_1"] = {}

    MaskDict["d2_neu_mo"]["column"] = ["neu_mo_mask", "Time point"]
    MaskDict["d2_neu_mo"]["criteria"] = [True, 2]

    MaskDict["fate_mask_d2"]["column"] = ["has_fate_mask", "Time point"]
    MaskDict["fate_mask_d2"]["criteria"] = [True, 2]

    MaskDict["early_d2_neu_mo"]["column"] = ["Time point", "neu_mo_mask", "early_cells"]
    MaskDict["early_d2_neu_mo"]["criteria"] = [2, True, True]

    MaskDict["d2_neu_mo_no_outliers"]["column"] = [
        "Time point",
        "neu_mo_mask",
        "outliers",
    ]
    MaskDict["d2_neu_mo_no_outliers"]["criteria"] = [2, True, False]

    MaskDict["neu_mo_no_outliers"]["column"] = ["neu_mo_mask", "outliers"]
    MaskDict["neu_mo_no_outliers"]["criteria"] = [True, False]

    MaskDict["neu_mo"]["column"] = ["neu_mo_mask"]
    MaskDict["neu_mo"]["criteria"] = [True]
    
    MaskDict["fate_mask_d2"]["column"] = ["has_fate_mask", "Time point"]
    MaskDict["fate_mask_d2"]["criteria"] = [True, 2]
    
    MaskDict["filtered_fate_heldout_0"]["column"] = ["has_fate_mask", "Time point", "heldout_mask",]
    MaskDict["filtered_fate_heldout_0"]["criteria"] = [True, 2, 0]
    MaskDict["filtered_fate_heldout_1"]["column"] = ["has_fate_mask", "Time point", "heldout_mask",]
    MaskDict["filtered_fate_heldout_1"]["criteria"] = [True, 2, 1]
    MaskDict["filtered_fate_early_0"]["column"] = ["has_fate_mask", "Time point", "early_neu_mo",]
    MaskDict["filtered_fate_early_0"]["criteria"] = [True, 2, 0]
    MaskDict["filtered_fate_early_1"]["column"] = ["has_fate_mask", "Time point", "early_neu_mo",]
    MaskDict["filtered_fate_early_1"]["criteria"] = [True, 2, 1]
    MaskDict["filtered_fate_heldout_0_early_0"]["column"] = ["has_fate_mask", "Time point", "heldout_mask", "early_neu_mo",]
    MaskDict["filtered_fate_heldout_0_early_0"]["criteria"] = [True, 2, 0, 0]
    MaskDict["filtered_fate_heldout_1_early_0"]["column"] = ["has_fate_mask", "Time point", "heldout_mask", "early_neu_mo",]
    MaskDict["filtered_fate_heldout_1_early_0"]["criteria"] = [True, 2, 1, 0]
    MaskDict["filtered_fate_heldout_0_early_1"]["column"] = ["has_fate_mask", "Time point", "heldout_mask", "early_neu_mo",]
    MaskDict["filtered_fate_heldout_0_early_1"]["criteria"] = [True, 2, 0, 1]
    MaskDict["filtered_fate_heldout_1_early_1"]["column"] = ["has_fate_mask", "Time point", "heldout_mask", "early_neu_mo",]
    MaskDict["filtered_fate_heldout_1_early_1"]["criteria"] = [True, 2, 1, 1]

    return MaskDict