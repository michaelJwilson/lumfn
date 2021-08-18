import numpy as np

def tmr_ecorr(tt, zz, zref=0.0, band='r'):
    types = {'gray': 0.97, 'blue': 2.12, 'red': 0.80}

    return  -1. * types[tt] * (zz - zref)

    
