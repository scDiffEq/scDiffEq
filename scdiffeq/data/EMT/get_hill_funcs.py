from .get_Mu import Mu
from .get_zs import get_z, get_s

def get_hill_functions(kmrgd, degredation, transcription_rate, hill_threshold, cooperativity, fold_change):
    
    Mu0, Mu1, Mu2, Mu3, Mu4, Mu5, Mu6, Mu20, Mu21, Mu22 = Mu(kmrgd, hill_threshold.u2000, cooperativity.nu200, hill_threshold.u340, cooperativity.nu34)

    z = get_z(transcription_rate.gz, kmrgd, Mu0, Mu1, Mu2, Mu3, Mu4, Mu5, Mu6, degredation.kz)
    s = get_s(transcription_rate.gs, kmrgd, Mu20, Mu21, Mu22, degredation.ks)
    
    class _get_hill_funcs:
        
        def __init__(self, Hillszu200, Hillssu200, Hillszmz, Hillssmz, Hillssu34, Hillszu34, Hillssms, Hillsi2ms):
            self.Hillszu200 = Hillszu200
            self.Hillssu200 = Hillssu200
            self.Hillszmz = Hillszmz
            self.Hillssmz = Hillssmz
            self.Hillssu34 = Hillssu34
            self.Hillszu34 = Hillszu34
            self.Hillssms = Hillssms
            self.Hillsi2ms = Hillsi2ms
            
    class _inheret_:
        
        def __init__(self, degredation, transcription_rate, hill_threshold, cooperativity, fold_change, Mu0, Mu1, Mu2, Mu3, Mu4, Mu5, Mu6, Mu20, Mu21, Mu22):
            
            self.degredation, self.transcription_rate, self.hill_threshold, self.cooperativity, self.fold_change = degredation, transcription_rate, hill_threshold, cooperativity, fold_change
         
            self.Mu0, self.Mu1, self.Mu2, self.Mu3, self.Mu4, self.Mu5, self.Mu6, self.Mu20, self.Mu21, self.Mu22 = Mu0, Mu1, Mu2, Mu3, Mu4, Mu5, Mu6, Mu20, Mu21, Mu22

    Hillszu200 = (1 + fold_change.lamdazu200 * (z / hill_threshold.z0u200) ** cooperativity.nzu200) / (
        1 + (z / hill_threshold.z0u200) ** cooperativity.nzu200
    )
    Hillssu200 = (1 + fold_change.lamdasu200 * (s / hill_threshold.s0u200) ** cooperativity.nsu200) / (
        1 + (s / hill_threshold.s0u200) ** cooperativity.nsu200
    )
    Hillszmz = (1 + fold_change.lamdazmz * (z / hill_threshold.z0mz) ** cooperativity.nzmz) / (1 + (z / hill_threshold.z0mz) ** cooperativity.nzmz)
    Hillssmz = (1 + fold_change.lamdasmz * (s / hill_threshold.s0mz) ** cooperativity.nsmz) / (1 + (s / hill_threshold.s0mz) ** cooperativity.nsmz)
    Hillssu34 = (1 + fold_change.lamdasu34 * (s / hill_threshold.s0u34) ** cooperativity.nsu34) / (1 + (s / hill_threshold.s0u34) ** cooperativity.nsu34)
    Hillszu34 = (1 + fold_change.lamdazu34 * (z / hill_threshold.z0u34) ** cooperativity.nzu34) / (1 + (z / hill_threshold.z0u34) ** cooperativity.nzu34)
    Hillssms = (1 + fold_change.lamdasms * (s / hill_threshold.s0ms) ** cooperativity.nsms) / (1 + (s / hill_threshold.s0ms) ** cooperativity.nsms)
    Hillsi2ms = (1 + fold_change.lamdai2ms * (kmrgd[4] / hill_threshold.i20ms) ** cooperativity.ni2ms) / (
        1 + (kmrgd[4] / hill_threshold.i20ms) ** cooperativity.ni2ms
    )
    
    
    hill_funcs = _get_hill_funcs(Hillszu200, Hillssu200, Hillszmz, Hillssmz, Hillssu34, Hillszu34, Hillssms, Hillsi2ms)
    inhereted = _inheret_(degredation, transcription_rate, hill_threshold, cooperativity, fold_change, Mu0, Mu1, Mu2, Mu3, Mu4, Mu5, Mu6, Mu20, Mu21, Mu22)
    
    return hill_funcs, inhereted