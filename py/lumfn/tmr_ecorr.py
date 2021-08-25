import numpy as np

def tmr_ecorr(zz, tt='gray', zref=0.0, band='r'):
    # eqn. 2 of https://arxiv.org/pdf/1409.4681.pdf
    # Note: Galaxies are assumed to have no difference
    # in luminosity evolution between the r - and g-bands
    # when rest frame colours are calculated.  See pg. 8.
    types = {'gray': 0.97, 'blue': 2.12, 'red': 0.80}

    assert np.isin(band, ['g', 'r'])
    
    return  -1. * types[tt] * (zz - zref)

    
