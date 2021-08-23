import numpy             as     np
import pylab             as     pl 
import matplotlib.pyplot as     plt

from   distances         import comoving_distance, dist_mod, comoving_volume
from   ajs_kcorr         import ajs_kcorr
from   abs_mag           import abs_mag, app_mag
from   scipy.optimize    import brentq, minimize
from   params            import params
from   ref_gmr           import reference_gmr

def zmax(kcorr, rlim, Mh, obs_gmr, redshift, band='r', ref_z=params['ref_z'], lolim=0.01, hilim=0.7, ref_gmr=None, distance_only=False):
    if ref_gmr == None:
        ref_gmr = reference_gmr(kcorr, obs_gmr, redshift, zref=ref_z)
    
    def diff(zzmax):
        # Brent relies on sign difference above and below zero point.
        return (app_mag(kcorr, Mh, obs_gmr, zzmax, band=band, ref_z=ref_z, ref_gmr=ref_gmr, distance_only=distance_only).item() - rlim)

    def absdiff(zz):
        return np.abs(diff(zz))

    # print(redshift, diff(lolim), diff(hilim))

    try:
        result, rR = brentq(diff, lolim, hilim, disp=True, full_output=True)
    
    except ValueError as E:
        # print('Failed on {} with zmax of {} ({}, {})'.format(redshift, rR.x0, rR.iterations, rR.flag))
        
        # print('Falling back to minimizer for Mrh: {}'.format(Mrh))
        
        # If sufficiently bright, not fainter than rlim at hilim.
        # Brent method fails, requires sign change across boundaries. 
        result = minimize(absdiff, redshift, method='Nelder-Mead')
        
        if result.success:
            result = result.x[0]
        #
        else:
            print(result.message)
            
            raise RuntimeError()
              
    return  result

def vmax(kcorr, rlim, Mh, obs_gmr, zz, band='r', ref_z=params['ref_z'], min_z=1.e-16, fsky=1.0, max_z=None, distance_only=False):
    if max_z == None:
        max_z = zmax(kcorr, rlim, Mh, obs_gmr, zz, band='r', ref_z=ref_z, distance_only=distance_only)
    
    return  comoving_volume(min_z, max_z, fsky)

     
if __name__ == '__main__':
     x           = ajs_kcorr()

     rlim        = 19.8
     
     Mh          = -21.
     redshift    = 0.30
     obs_gmr     = 1.50

     fig, axes   = plt.subplots(1, 3, figsize=(20, 6))

     axes[0].axhline(19.8, c='k', lw=0.1)

     ref_gmr     = reference_gmr(x, obs_gmr, redshift, zref=0.0)

     zs          = np.arange(0.01, 0.7, 0.01)

     colors      = plt.rcParams['axes.prop_cycle'].by_key()['color']

     for distance_only, alpha in zip([True, False], [0.5, 1.0]):
          mm     = np.array([app_mag(x, Mh, obs_gmr, zz, ref_gmr=ref_gmr, distance_only=distance_only).item() for zz in zs])
          res    = np.array([zmax(x, rlim, Mh, obs_gmr, zz, distance_only=distance_only) for zz in zs])
          VV     = np.array([vmax(x, rlim, Mh, obs_gmr, zz, distance_only=distance_only) for zz in zs])

          axes[0].plot(zs, mm, c=colors[0], lw=1., alpha=alpha)
          axes[1].plot(zs, res, c=colors[1], lw=1., alpha=alpha)
          axes[2].plot(zs, VV / 1.e9, c=colors[2], lw=1., alpha=alpha)

     for ax in axes:
         ax.set_xlabel(r'$z$')

     axes[0].set_ylabel('app. $r-$band mag.')
     axes[1].set_ylabel('$z$ max')
     axes[2].set_ylabel('$V$ max / 1.e9')

     fig.suptitle('$M_r = {:.2f}$ with obs. ($g-r$) = {:.2f} at $z={:.2f}$'.format(Mh, obs_gmr, redshift))

     # plt.tight_layout()
     
     pl.show()
         
     # pl.clf()
     # pl.plot(Ms, res)
     # pl.xlabel('M')
     # pl.ylabel('zmax')
     # pl.show()
