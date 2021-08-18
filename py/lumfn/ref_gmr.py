import scipy
import numpy          as     np
import pylab          as     pl 

from   distances      import comoving_distance, dist_mod

from   ajs_kcorr      import ajs_kcorr

from   tmr_kcorr      import tmr_kcorr
from   tmr_ecorr      import tmr_ecorr
from   scipy.optimize import brentq, minimize
from   params         import params

def _reference_gmr(kcorr, gmr, z, zref=params['ref_z']):
     # Calling signature: ajs_kcorr.eval(self, ref_gmr, zz, band, ref_z=0.1)
     def diff(x):
          # Here, x is the ref_color to be solved for. 
          obs  = x - kcorr.eval(x, zref, band='g', ref_z=zref) + kcorr.eval(x, zref, band='r', ref_z=zref)
          obs += kcorr.eval(x, z, band='g', ref_z=zref) - kcorr.eval(x, z, band='r', ref_z=zref)

          return (gmr - obs)

     def absdiff(x):
        return np.abs(diff(x))
   
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

def reference_gmr(kcorr, gmrs, zs, zref=params['ref_z']):
     result = []

     zs     = np.atleast_1d(np.array(zs, copy=True))
     gmrs   = np.atleast_1d(np.array(gmrs, copy=True))
     
     for gmr in gmrs:
          row = []
          
          for zz in zs:
               row.append(_reference_gmr(kcorr, gmr, zz, zref=params['ref_z']))

          result.append(row)

     return  np.array(result)


if __name__ == '__main__':
     kcorr   = ajs_kcorr()

     zs      = np.arange(0.1, 0.4, 0.1)
     gmrs    = np.arange(1.25, 1.75, 0.25) 
     
     ref_gmr = reference_gmr(kcorr, gmrs, zs, zref=0.1)

     print(ref_gmr)
