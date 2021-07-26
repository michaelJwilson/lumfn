import numpy          as     np
import pylab          as     pl 

from   distances      import comoving_distance, dist_mod
from   tmr_kcorr      import tmr_kcorr
from   tmr_ecorr      import tmr_ecorr
from   abs_mag        import abs_mag, app_mag
from   scipy.optimize import brentq


def zmax(kcorr, Mrh, rlim):
     def diff(zz):
          mm  = app_mag(kcorr, Mrh, zz)

          return rlim - mm

     return  brentq(diff, 0.0, 0.6)

def vmax(kcorr, Mrh, rlim, min_z=0.0, fsky=1.0):
     max_z = zmax(kcorr, Mrh, rlim)

     xmax3 = comoving_distance(max_z) ** 3.
     xmin3 = comoving_distance(min_z) ** 3.

     return (4. * np.pi * fsky / 3.) * (xmax3 - xmin3)
     
     
if __name__ == '__main__':
     Ms  = np.arange(-22., -19., 0.5)

     x   = tmr_kcorr()
     
     res = np.array([zmax(x, M, 19.8) for M in Ms])
     Vs  = np.array([vmax(x, M, 19.8) for M in Ms])
     
     pl.clf()
     pl.plot(Ms, res)
     pl.xlabel('M')
     pl.ylabel('zmax')
     pl.show()
