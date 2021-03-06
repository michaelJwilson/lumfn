import scipy
import numpy          as     np
import pylab          as     pl 

from   distances      import comoving_distance, dist_mod
from   scipy.optimize import brentq, minimize


def ajs_reference_gmr(kcorr, gmr, z):
     assert kcorr.z0 ==	0.1
     
     def diff(x):
          # Here, x is the ref_color to be solved for in the native
          # reference, i.e. z=0.1 for AJS. 
          obs  = x
                    
          obs += kcorr.ref_eval(x, z, band='g')
          obs -= kcorr.ref_eval(x, z, band='r')

          return (gmr - obs)

     def absdiff(x):
        return np.abs(diff(x))
   
     try:
        # rest color limits.  
        result = brentq(diff, -3., 3.)

     except ValueError as VE:
        # Brent method fails, requires sign change across boundaries.                                                                                                                                                      
        result = minimize(absdiff, 0.75)

        if result.success:
            result = result.x[0]

        else:             
            print(result.message)

            raise RuntimeError()

     return  result

def tmr_reference_gmr(kcorr, ajs_ref_gmr):
     assert kcorr.z0 == 0.1
     
     # E has no affect on colors (assumption).
     shift  = kcorr.ref_eval(ajs_ref_gmr, 0.0, band='g') - kcorr.ref_eval(ajs_ref_gmr, 0.0, band='r')

     return ajs_ref_gmr + shift
     
def reference_gmr(kcorr, gmrs, zs):
     result = []

     zs     = np.atleast_1d(np.array(zs, copy=True))
     gmrs   = np.atleast_1d(np.array(gmrs, copy=True))
     
     for gmr in gmrs:
          row = []
          
          for zz in zs:
               row.append(ajs_reference_gmr(kcorr, gmr, zz))

          result.append(row)

     return  np.array(result)


if __name__ == '__main__':
     from   ajs_kcorr      import ajs_kcorr
     from   tmr_kcorr      import tmr_kcorr

     
     x       = ajs_kcorr()  # [ajs_kcorr, tmr_kcorr]                                                                                                                                                           
     zz      =   0.2

     rp      =  19.0
     band    =    'r'

     obs_gmr = 1.000

     for zz in np.arange(0.01, 0.5, 0.005):
          ref_gmr = one_reference_gmr(x, obs_gmr, zz, zref=params['ref_z'])
          
          pl.plot(zz, ref_gmr, marker='.', c='k')

     pl.xlabel(r'$z$')
     pl.ylabel(r'ref. $(g-r)$')
     
     pl.show()
