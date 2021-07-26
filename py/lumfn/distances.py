import numpy as np
import pylab as pl

from   params            import params 
from   astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=100. * params['h'], Om0=params['Om'], Tcmb0=2.725)

def comoving_distance(zz=0.1):
    return params['h'] * cosmo.luminosity_distance(zz).value

def dist_mod(chi):
    return 5. * np.log10(chi) + 25.

if __name__ == '__main__':
    zs  = np.arange(0.0, 2.0, 0.01)
    
    xs  = comoving_distance(zz=zs) # [Mpc/h]
    mus = dist_mod(xs)

    pl.plot(zs, mus)
    pl.show()
