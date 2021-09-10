import numpy as np
import pylab as pl

from   params            import params 
from   astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=100. * params['h'], Om0=params['Om'], Tcmb0=2.725)

def comoving_distance(zz):
    return params['h'] * cosmo.comoving_distance(zz).value

def dist_mod(zz):
    lumdist = params['h'] * cosmo.luminosity_distance(zz).value

    return 5. * np.log10(lumdist) + 25.
    
def comoving_volume(min_z, max_z, fsky):
     xmin3 = comoving_distance(min_z) ** 3.
     xmax3 = comoving_distance(max_z) ** 3.
     
     return (4. * np.pi * fsky / 3.) * (xmax3 - xmin3)

if __name__ == '__main__':
    zs  = np.arange(0.0, 2.0, 0.01)
    
    xs  = comoving_distance(zz=zs) # [Mpc/h]
    mus = dist_mod(zs)

    pl.plot(zs, mus)
    pl.show()
