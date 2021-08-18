import scipy
import numpy          as     np
import pylab          as     pl 

from   distances      import comoving_distance, dist_mod

from   ajs_kcorr      import ajs_kcorr

from   tmr_kcorr      import tmr_kcorr
from   tmr_ecorr      import tmr_ecorr
from   scipy.optimize import brentq, minimize
from   params         import params


def ref_gmr(kcorr, gmr, z, zref=params['ref_z']):
     def diff(x):
          # Here, x is the ref_color to be solved for. 
          obs  = x - kcorr.eval(x, zref, band='g') + kcorr.eval(x, zref, band='r')
          obs += kcorr.eval(x, z, band='g') - kcorr.eval(x, z, band='r')

          return (gmr - obs)

     def absdiff(x):
        return np.abs(diff(x))

     assert (kcorr.z0 == zref), 'zref={:.2f} required for this kcorrection.'.format(kcorr.z0)
   
     try:
        # rest color limits.  
        result = brentq(diff, -3., 3.)

     except ValueError as VE:
        # Brent method fails, requires sign change across boundaries.                                                                                                                                                      
        result = minimize(absdiff, 0.5)

        if result.success:
            result = result.x[0]

        else:
            print(result.message)

            raise RuntimeError()

     return  result


if __name__ == '__main__':
     kcorr = ajs_kcorr()

     ref_gmr = ref_gmr(kcorr, gmr=1.5, z=0.4, zref=0.1)

     print(ref_gmr)
