import scipy
import numpy     as     np
import pylab     as     pl 

from   distances import comoving_distance, dist_mod

from   ajs_kcorr import ajs_kcorr

from   params    import params

from   tmr_kcorr import tmr_kcorr
from   tmr_ecorr import	tmr_ecorr
from   ref_gmr   import reference_gmr


def abs_mag(kcorr, rpetro, ref_gmr, zz, band='r', ref_z=params['ref_z'], distance_only=False):
     chi = comoving_distance(zz)
     mu  = dist_mod(chi)

     # Note: tmr references to z=0.0; ajs to z=0.1;
     kk  = kcorr.eval(ref_gmr, zz, band=band, ref_z=ref_z)

     # Note:  defaults to gray type. 
     EE  = tmr_ecorr(zz, band=band)
     
     #  Returns:  M - 5log10(h)
     #  See https://arxiv.org/pdf/1409.4681.pdf

     if distance_only:
          return rpetro - mu
     else:
          return rpetro - mu - kk - EE

def app_mag(kcorr, Mh, obs_gmr, zz, band='r', ref_z=params['ref_z'], ref_gmr=None, distance_only=False):     
     # Mh \equiv Mr - 5log10(h)
     chi      = comoving_distance(zz)
     mu       = dist_mod(chi)

     if ref_gmr == None:
          ref_gmr = reference_gmr(kcorr, obs_gmr, zz, zref=ref_z)

     kk       = kcorr.eval(ref_gmr, zz, band=band, ref_z=ref_z)

     # Note:  defaults to gray type.
     EE       = tmr_ecorr(zz, band=band)

     if distance_only:
          return Mh + mu
     else:
          return Mh + mu + kk + EE


if __name__ == '__main__':
     x       = ajs_kcorr()  # [ajs_kcorr, tmr_kcorr]

     zz      =   0.4
     
     rp      =  19.8
     Mh      = -21.0

     ref_gmr = 0.708
     obs_gmr = 0.500 
     
     MM      = abs_mag(x, rp, ref_gmr, zz,  band='r')
     
     for zz in np.arange(0.01, 0.5, 0.025):
          ref_gmr = reference_gmr(x, obs_gmr, zz, zref=params['ref_z'])
          mm      = app_mag(x, Mh, obs_gmr, zz, band='r').item()

          pl.plot(zz, ref_gmr, marker=',', lw=0.0)
          # pl.plot(zz, mm, marker=',', lw=0.0)
     
     # pl.plot(zs, MM)
     # pl.xlabel('z')
     # pl.ylabel('M')
     # pl.show()
     
     # pl.clf()
     # pl.plot(zs, mm)
     pl.xlabel('z')
     pl.ylabel('ref. $g-r$')
     # pl.ylabel('r')
     pl.show()
