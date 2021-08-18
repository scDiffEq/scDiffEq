def Mu(kmrgd, u2000, nu200, u340, nu34):

    Mu0 = 1 / (1 + kmrgd[0] / u2000) ** nu200
    Mu1 = (kmrgd[0] / u2000) / (1 + kmrgd[0] / u2000) ** nu200
    Mu2 = (kmrgd[0] / u2000) ** 2 / (1 + kmrgd[0] / u2000) ** nu200
    Mu3 = (kmrgd[0] / u2000) ** 3 / (1 + kmrgd[0] / u2000) ** nu200
    Mu4 = (kmrgd[0] / u2000) ** 4 / (1 + kmrgd[0] / u2000) ** nu200
    Mu5 = (kmrgd[0] / u2000) ** 5 / (1 + kmrgd[0] / u2000) ** nu200
    Mu6 = (kmrgd[0] / u2000) ** 6 / (1 + kmrgd[0] / u2000) ** nu200

    Mu20 = 1 / (1 + kmrgd[2] / u340) ** nu34
    Mu21 = (kmrgd[2] / u340) / (1 + kmrgd[2] / u340) ** nu34
    Mu22 = (kmrgd[2] / u340) ** 2 / (1 + kmrgd[2] / u340) ** nu34

    return Mu0, Mu1, Mu2, Mu3, Mu4, Mu5, Mu6, Mu20, Mu21, Mu22
