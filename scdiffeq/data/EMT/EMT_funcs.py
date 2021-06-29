import numpy as np
from .set_params import set_parameters
from .get_hill_funcs import get_hill_functions
from .get_dydt import get_miR200, get_mZ, get_miR34, get_mS, get_i2

##########################################
    

def EMT_dydt(kmrgd, t, parameters):
    
    """Get miR200, mZ, miR34, mS, and i2"""
    
    p = parameters
    
    hill, _ = get_hill_functions(kmrgd, p.degredation, p.transcription_rate, p.hill_threshold, p.cooperativity, p.fold_change)
    
    miR200 = get_miR200(_.transcription_rate.gu200, kmrgd, _.Mu1, _.Mu2, _.Mu3, _.Mu4, _.Mu5, _.Mu6, _.degredation.ku200, hill.Hillszu200, hill.Hillssu200)
    mZ = get_mZ(_.transcription_rate.gmz, hill.Hillszmz, hill.Hillssmz, kmrgd, _.Mu1, _.Mu2, _.Mu3, _.Mu4, _.Mu5, _.Mu6, _.degredation.kmz)
    miR34 = get_miR34(_.transcription_rate.gu34, hill.Hillssu34, hill.Hillszu34, kmrgd, _.Mu21, _.Mu22, _.degredation.ku34)
    mS = get_mS(_.transcription_rate.gms, hill.Hillssms, hill.Hillsi2ms, kmrgd, _.Mu21, _.Mu22, _.degredation.kms)
    i2 = get_i2(t)
    
    return np.array([miR200, mZ, miR34, mS, i2])
    
def parameterize(variance):

    (
        degredation,
        transcription_rate,
        hill_threshold,
        cooperativity,
        fold_change,
    ) = set_parameters(variance)

    class parameters:
        def __init__(
            self,
            degredation,
            transcription_rate,
            hill_threshold,
            cooperativity,
            fold_change,
        ):

            self.degredation = degredation
            self.transcription_rate = transcription_rate
            self.hill_threshold = hill_threshold
            self.cooperativity = cooperativity
            self.fold_change = fold_change

    params = parameters(
        degredation, transcription_rate, hill_threshold, cooperativity, fold_change
    )

    return params
    