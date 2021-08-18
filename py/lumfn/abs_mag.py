import scipy
import numpy     as     np
import pylab     as     pl 

from   distances import comoving_distance, dist_mod

from   ajs_kcorr import ajs_kcorr

from   params    import params

from   tmr_kcorr import tmr_kcorr
from   tmr_ecorr import	tmr_ecorr
from   ref_gmr   import reference_gmr


def abs_mag(kcorr, rpetro, ref_gmr, zz, band='r', ref_z=params['ref_z']):
     chi = comoving_distance(zz)
     mu  = dist_mod(chi)

     # Note: tmr references to z=0.0; ajs to z=0.1;
     kk  = kcorr.eval(ref_gmr, zz, band=band, ref_z=ref_z)

     # TODO: Base on ref_gmr.
     tt  = 'gray'

     # Note: tmr references to z=0.0.
     EE  = tmr_ecorr(tt, zz, band=band)
     
     #  Returns:  M - 5log10(h)
     #  See https://arxiv.org/pdf/1409.4681.pdf
     return  rpetro - mu - kk - EE

def app_mag(kcorr, Mh, gmr, zz, band='r', ref_z=params['ref_z']):     
     # Mh \equiv Mr - 5log10(h)
     chi      = comoving_distance(zz)
     mu       = dist_mod(chi)

     iref_gmr = reference_gmr(kcorr, gmr, zz, zref=ref_z)

     kk       = kcorr.eval(iref_gmr, zz, band=band, ref_z=ref_z)
                       
     tt       = 'gray'  # Base on ref_gmr.                                                                                                                                                                                   
     EE       = tmr_ecorr(tt, zz)
     
     return  Mh + mu + kk + EE


if __name__ == '__main__':
     x   = ajs_kcorr()  # [ajs_kcorr, tmr_kcorr]
     
     zs  = np.arange(0.1, 0.6, 0.1)
     
     MM  = abs_mag(x, 19.8, 0.708, zs,  band='r')
     mm  = app_mag(x, -21., 0.708, 0.2, band='r')
     
     # pl.plot(zs, MM)
     # pl.xlabel('z')
     # pl.ylabel('M')
     # pl.show()
     
     # pl.clf()
     # pl.plot(zs, mm)
     # pl.xlabel('z')
     # pl.ylabel('r')
     # pl.show()
