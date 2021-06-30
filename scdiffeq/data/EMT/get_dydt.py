def get_miR200(gu200, kmrgd, Mu1, Mu2, Mu3, Mu4, Mu5, Mu6, ku200, Hillszu200, Hillssu200):
        
    miR200 = (
        gu200 * Hillszu200 * Hillssu200
        - kmrgd[1]
        * (
            0.005 * 6 * Mu1
            + 2 * 0.05 * 15 * Mu2
            + 3 * 0.5 * 20 * Mu3
            + 4 * 0.5 * 15 * Mu4
            + 5 * 0.5 * 6 * Mu5
            + 6 * 0.5 * Mu6
        )
        - ku200 * kmrgd[0]
    )
    
    return miR200


def get_mZ(gmz, Hillszmz, Hillssmz, kmrgd, Mu1, Mu2, Mu3, Mu4, Mu5, Mu6, kmz):    
    
    mZ = (
        gmz * Hillszmz * Hillssmz
        - kmrgd[1]
        * (0.04 * 6 * Mu1 + 0.2 * 15 * Mu2 + 20 * Mu3 + 15 * Mu4 + 6 * Mu5 + Mu6)
        - kmz * kmrgd[1]
    )
    
    return mZ

def get_miR34(gu34, Hillssu34, Hillszu34, kmrgd, Mu21, Mu22, ku34):

    miR34 = (
        gu34 * Hillssu34 * Hillszu34
        - kmrgd[3] * (0.005 * 2 * Mu21 + 2 * 0.05 * Mu22)
        - ku34 * kmrgd[2]
    )
    
    return miR34


def get_mS(gms, Hillssms, Hillsi2ms, kmrgd, Mu21, Mu22, kms):
    
    mS = (
        gms * Hillssms * Hillsi2ms
        - kmrgd[3] * (0.04 * 2 * Mu21 + 0.2 * Mu22)
        - kms * kmrgd[3]
    )
    
    return mS

def get_i2(t):
    
    if t < 150 * 24:
        i2 = 100000 / 150 / (24)
    else:
        i2 = -100000 / 150 / (24)
       
    return i2