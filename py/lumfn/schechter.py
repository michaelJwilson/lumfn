import numpy as np

def schechter(M, log10phistar=-2.01, Mstar=-20.89, alpha=-1.25):
    phistar = 10. ** log10phistar

    expa    = 10. ** (0.4 * (Mstar - M) * (1. + alpha))
    expb    = np.exp(-10. ** (0.4 * (Mstar - M)))

    return  np.log(10.) * phistar * expa * expb / (2.5)
