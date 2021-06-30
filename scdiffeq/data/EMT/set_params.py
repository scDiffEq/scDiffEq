import numpy as np

def set_parameters(variance):
    
    ### defaults:

    # Degradation rate:
    ku200, kmz, kz, ku34, kms, ks = 0.05, 0.5, 0.1, 0.05, 0.5, 0.125
    # transcription rate:
    gu200, gmz, gz, gu34, gms, gs = 2100, 11, 100, 1350, 90, 100

    # Hill Threshold
    z0u200, z0mz, s0u200, s0mz = 220000, 25000, 180000, 180000
    u2000, u340, s0u34, z0u34 = 10000, 10000, 300000, 600000
    s0ms, i20ms = 200000, 50000

    # Cooperativity
    nzu200, nsu200, nzmz, nsmz, nu200 = 3, 2, 2, 2, 6
    nu34, nsu34, nzu34, nsms, ni2ms = 2, 1, 1, 1, 1

    # fold change
    lamdazu200, lamdasu200, lamdazmz, lamdasmz = 0.1, 0.1, 7.5, 10
    lamdasu34, lamdazu34, lamdasms, lamdai2ms = 0.1, 1, 0.1, 10

    ###

    class get_degredation:
        def __init__(self, ku200, kmz, kz, ku34, kms, ks, variance):
            
            if variance != None:
                variance = abs(np.random.normal(0, variance, 6) + 1)
                
            else:
                variance = np.ones(6)
            
            self.ku200 = ku200 * variance[0]
            self.kmz = kmz * variance[1]
            self.kz = kz * variance[2]
            self.ku34 = ku34 * variance[3]
            self.kms = kms * variance[4]
            self.ks = ks * variance[5]

    class get_transcription_rate:
        def __init__(self, gu200, gmz, gz, gu34, gms, gs, variance):
            
            if variance != None:
                variance = abs(np.random.normal(0, variance, 6) + 1)
            else:
                variance = np.ones(6)
                
            self.gu200 = gu200 * variance[0]
            self.gmz = gmz * variance[1]
            self.gz = gz * variance[2]
            self.gu34 = gu34 * variance[3]
            self.gms = gms * variance[4]
            self.gs = gs * variance[5]

    class get_hill_threshold:
        def __init__(
            self,
            z0u200,
            z0mz,
            s0u200,
            s0mz,
            u2000,
            u340,
            s0u34,
            z0u34,
            s0ms,
            i20ms,
            variance,
        ):
            if variance != None:
                variance = abs(np.random.normal(0, variance, 10) + 1)
            else:
                variance = np.ones(10)
            
            self.z0u200 = z0u200 * variance[0]
            self.z0mz = z0mz * variance[1]
            self.s0u200 = s0u200 * variance[2]
            self.s0mz = s0mz * variance[3]
            self.u2000 = u2000 * variance[4]
            self.u340 = u340 * variance[5]
            self.s0u34 = s0u34 * variance[6]
            self.z0u34 = z0u34 * variance[7]
            self.s0ms = s0ms * variance[8]
            self.i20ms = i20ms * variance[9]

    class get_cooperativity:
        def __init__(
            self,
            nzu200,
            nsu200,
            nzmz,
            nsmz,
            nu200,
            nu34,
            nsu34,
            nzu34,
            nsms,
            ni2ms,
            variance,
        ):
            
            if variance != None:
                variance = abs(np.random.normal(0, variance, 10) + 1)
            else:
                variance = np.ones(10)
                
            self.nzu200 = nzu200 * variance[0]
            self.nsu200 = nsu200 * variance[1]
            self.nzmz = nzmz * variance[2]
            self.nsmz = nsmz * variance[3]
            self.nu200 = nu200 * variance[4]
            self.nu34 = nu34 * variance[5]
            self.nsu34 = nsu34 * variance[6]
            self.nzu34 = nzu34 * variance[7]
            self.nsms = nsms * variance[8]
            self.ni2ms = ni2ms * variance[9]

    class get_fold_change:
        def __init__(
            self,
            lamdazu200,
            lamdasu200,
            lamdazmz,
            lamdasmz,
            lamdasu34,
            lamdazu34,
            lamdasms,
            lamdai2ms,
            variance,
        ):
            if variance != None:
                variance = abs(np.random.normal(0, variance, 8) + 1)
            else:
                variance = np.ones(8)
                
            self.lamdazu200 = lamdazu200 * variance[0]
            self.lamdasu200 = lamdasu200 * variance[1]
            self.lamdazmz = lamdazmz * variance[2]
            self.lamdasmz = lamdasmz * variance[3]
            self.lamdasu34 = lamdasu34 * variance[4]
            self.lamdazu34 = lamdazu34 * variance[5]
            self.lamdasms = lamdasms * variance[6]
            self.lamdai2ms = lamdai2ms * variance[7]

    degredation = get_degredation(ku200, kmz, kz, ku34, kms, ks, variance)
    transcription_rate = get_transcription_rate(gu200, gmz, gz, gu34, gms, gs, variance)
    hill_threshold = get_hill_threshold(
        z0u200, z0mz, s0u200, s0mz, u2000, u340, s0u34, z0u34, s0ms, i20ms, variance
    )
    cooperativity = get_cooperativity(
        nzu200, nsu200, nzmz, nsmz, nu200, nu34, nsu34, nzu34, nsms, ni2ms, variance
    )
    fold_change = get_fold_change(
        lamdazu200,
        lamdasu200,
        lamdazmz,
        lamdasmz,
        lamdasu34,
        lamdazu34,
        lamdasms,
        lamdai2ms,
        variance,
    )    

    return degredation, transcription_rate, hill_threshold, cooperativity, fold_change