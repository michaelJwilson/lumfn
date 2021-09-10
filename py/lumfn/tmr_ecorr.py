import numpy  as np

from   params import params

def tmr_ecorr(zz, tmr_ref_gmr, band='r'):
    # eqn. 2 of https://arxiv.org/pdf/1409.4681.pdf
    # Note: Galaxies are assumed to have no difference
    # in luminosity evolution between the r - and g-bands
    # when rest frame colours are calculated.  See pg. 8.
    types = {'gray': 0.97, 'blue': 2.12, 'red': 0.80}

    assert np.isin(band, ['g', 'r'])

    tt    = 'blue' if (tmr_ref_gmr <= params['rf_gmr_redblue']) else 'red'
    
    return  -1. * types[tt] * zz

    
