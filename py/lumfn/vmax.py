import numpy          as     np
import pylab          as     pl 

from   distances      import comoving_distance, dist_mod, comoving_volume
from   tmr_kcorr      import tmr_kcorr
from   tmr_ecorr      import tmr_ecorr
from   abs_mag        import abs_mag, app_mag
from   scipy.optimize import brentq, minimize


def zmax(kcorr, Mrh, rlim, lolim=0.0, hilim=3.0):
    def diff(zz):
        mm  = app_mag(kcorr, Mrh, zz)

        # Brent relies on sign differen above and below zero point.
        return rlim - mm

    def absdiff(zz):
        return np.abs(diff(zz))
    
    try:
        result = brentq(diff, lolim, hilim)
        
    except ValueError as VE:
        # print('Falling back to minimizer for Mrh: {}'.format(Mrh))
        
        # If sufficiently bright, not fainter than rlim at hilim.
        # Brent method fails, requires sign change across boundaries. 
        result = minimize(absdiff, 0.6)
        
        if result.success:
            result = result.x[0]
        
        else:
            print(result.message)
            
            raise RuntimeError()
                        
    return  result

def vmax(kcorr, Mrh, rlim, min_z=0.0, fsky=1.0):
    max_z = zmax(kcorr, Mrh, rlim)
    
    return  comoving_volume(min_z, max_z, fsky)

     
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
