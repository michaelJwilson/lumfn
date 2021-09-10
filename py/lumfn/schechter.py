import numpy as np
import scipy.integrate as integrate

from   data.schechters import schechters
from   scipy.special   import gamma
from   distances       import dist_mod
from   params          import params as main_params
from   zlimits         import kmax


def schechter(M, named_type='TMR', zz=None, only_params=False):
    params       = schechters[named_type]
    
    log10phistar = params['log10phistar']
    Mstar        = params['Mstar']
    alpha        = params['alpha'] 

    P            = params['P']
    Q            = params['Q']
    zref         = params['zref']

    zz           = np.array([zz], copy=True)[0]
    
    if zz == None:
        zz       = zref
        
    phistar      = 10. ** log10phistar

    # Evolution:
    Mstar       -= Q * (zz - zref)
    phistar     *= 10. ** (0.4 * P * (zz - zref))

    if only_params:
        return (Mstar, phistar, alpha)
    
    expa         = 10. ** (0.4 * (Mstar - M) * (1. + alpha))
    expb         = np.exp(-10. ** (0.4 * (Mstar - M)))

    return  np.log(10.) * phistar * expa * expb / 2.5

def schechter_prob(M, named_type='TMR', zz=None, Mmin=None, norm_only=False):
    params       = schechters[named_type]

    log10phistar = params['log10phistar']
    Mstar        = params['Mstar']
    alpha        = params['alpha']

    P            = params['P']
    Q            = params['Q']
    zref         = params['zref']

    zz           = np.array([zz], copy=True)[0]
    
    if zz == None:
        zz       = zref
    
    phistar      = 10. ** log10phistar

    # Evolution:                                                                                                                                                                                                           
    Mstar       -= Q * (zz - zref)
    phistar     *= 10. ** (0.4 * P * (zz - zref)) 
    
    expa         = 10. ** (0.4 * (Mstar - M) * (1. + alpha))
    expb         = np.exp(-10. ** (0.4 * (Mstar - M)))

    if Mmin == None:
        rlim     = main_params['rlim']
        Mmin     = rlim - dist_mod(zz) - kmax(zz)

    tmin         = 10.**(-0.4 * (Mmin - Mstar))

    kernel       = lambda t: np.exp(-t) * (t**alpha)
    norm, err    = integrate.quad(kernel, tmin, 100.)

    if norm_only:
        return tmin, norm, err, phistar

    else:
        return  np.log(10.) * expa * expb / 2.5 / norm


def mxxl_kernel(zz):
    return 1. / (1.  + np.exp(-100. * (zz - 0.15)))

def mxxl_phistar(zz):
    zz   = np.array([zz], copy=True)[0]

    if zz == None:
        zz = zref
    
    kern = mxxl_kernel(zz)

    # Assumes alpha equates to loveday alpha, as true of effective integrand
    # to integral over phi (dominated by large M).  
    blanton_phistar = 10. ** schechters['Blanton']['log10phistar']
    loveday_phistar = 10. ** schechters['LovedayMock']['log10phistar']

    blanton_P       = schechters['Blanton']['P']
    loveday_P       = schechters['LovedayMock']['P']
    
    return  kern * loveday_phistar * 10. ** (0.4 * loveday_P * (zz - 0.1)) + (1. - kern) * blanton_phistar * 10. ** (0.4 * blanton_P * (zz - 0.1))

def mxxl_schechter(M, zz):
    sdss = schechter(M, named_type='Blanton', zz=zz)
    gama = schechter(M, named_type='LovedayMock', zz=zz)

    kern = mxxl_kernel(zz)

    return kern * gama + (1. - kern) * sdss
