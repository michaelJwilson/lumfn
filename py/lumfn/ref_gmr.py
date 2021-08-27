import scipy
import numpy          as     np
import pylab          as     pl 

from   distances      import comoving_distance, dist_mod

from   ajs_kcorr      import ajs_kcorr

from   tmr_kcorr      import tmr_kcorr
from   scipy.optimize import brentq, minimize
from   params         import params

def one_reference_gmr(kcorr, gmr, z, zref=params['ref_z'], ecorr=True):
     assert  np.isin(zref, [0.0, 0.1])

     def diff(x):
          # Here, x is the ref_color to be solved for in the native
          # reference, i.e. z=0.1 for AJS. 
          obs  = x
                    
          obs += kcorr.ref_eval(x, z, band='g', ref_z=zref, ecorr=ecorr)
          obs -= kcorr.ref_eval(x, z, band='r', ref_z=zref, ecorr=ecorr)

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

     if zref == 0.0:
          shift   = kcorr.ref_eval(result, 0.0, band='g', ecorr=False) - kcorr.ref_eval(result, 0.0, band='r', ecorr=False)
          result += shift
       
     return  result

def reference_gmr(kcorr, gmrs, zs, zref=params['ref_z'], ecorr=True):
     result = []

     zs     = np.atleast_1d(np.array(zs, copy=True))
     gmrs   = np.atleast_1d(np.array(gmrs, copy=True))
     
     for gmr in gmrs:
          row = []
          
          for zz in zs:
               row.append(one_reference_gmr(kcorr, gmr, zz, zref=params['ref_z'], ecorr=ecorr))

          result.append(row)

     return  np.array(result)


if __name__ == '__main__':
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
