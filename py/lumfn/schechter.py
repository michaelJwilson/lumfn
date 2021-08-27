import numpy as np

from   data.schechters import schechters

def schechter(M, named_type='TMR'):
    params       = schechters[named_type]
    
    log10phistar = params['log10phistar']
    Mstar        = params['Mstar']
    alpha        = params['alpha'] 
    
    phistar = 10. ** log10phistar

    expa    = 10. ** (0.4 * (Mstar - M) * (1. + alpha))
    expb    = np.exp(-10. ** (0.4 * (Mstar - M)))

    return  np.log(10.) * phistar * expa * expb / (2.5)
