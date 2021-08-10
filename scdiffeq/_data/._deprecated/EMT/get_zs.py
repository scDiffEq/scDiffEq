def get_z(gz, kmrgd, Mu0, Mu1, Mu2, Mu3, Mu4, Mu5, Mu6, kz):
    
    z = (
        gz
        * kmrgd[1]
        * (
            Mu0
            + 0.6 * 6 * Mu1
            + 0.3 * 15 * Mu2
            + 0.1 * 20 * Mu3
            + 0.05 * 15 * Mu4
            + 0.05 * 6 * Mu5
            + 0.05 * Mu6
        )
        / kz
    )
    
    return z

def get_s(gs, kmrgd, Mu20, Mu21, Mu22, ks):
    
    s = gs * kmrgd[3] * (Mu20 + 0.6 * 2 * Mu21 + 0.3 * Mu22) / ks
    
    return s
    