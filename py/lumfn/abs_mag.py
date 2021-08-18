import scipy
import numpy     as     np
import pylab     as     pl 

from   distances import comoving_distance, dist_mod

from   ajs_kcorr import ajs_kcorr

from   tmr_kcorr import tmr_kcorr
from   tmr_ecorr import	tmr_ecorr


def abs_mag(kcorr, rpetro, ref_gmr, zz, band='r'):
     chi = comoving_distance(zz)
     mu  = dist_mod(chi)

     # Note: tmr references to z=0.0; ajs to z=0.1;
     kk  = kcorr.eval(ref_gmr, zz, band=band)

     # TODO: Base on ref_gmr.
     tt  = 'gray'

     # Note: tmr references to z=0.0.
     EE  = tmr_ecorr(tt, zz, band=band)
     
     #  Returns:  M - 5log10(h)
     #  See https://arxiv.org/pdf/1409.4681.pdf
     return  rpetro - mu - kk # - EE

def app_mag(kcorr, Mh, ref_gmr, zz, band='r'):
     # Mh \equiv Mr - 5log10(h)
     chi = comoving_distance(zz)
     mu  = dist_mod(chi)

     # TODO: predict ref_gmr given varying z. 
     kk  = kcorr.eval(ref_gmr, zz, band=band)

     tt  = 'gray'  # Base on ref_gmr.                                                                                                                                                                                   
     EE  = tmr_ecorr(tt, zz)
     
     return Mh + mu + kk # + E

# def app_gmr(kcorr, gmr, z, zp):
     # TODO: 

if __name__ == '__main__':
     # x = tmr_kcorr()
     x   = ajs_kcorr()
     
     zs  = np.arange(0.1, 0.6, 0.05)
     
     MM  = abs_mag(x, 19.8, 0.708, zs, band='r')
     mm  = app_mag(x, -21., 0.708, zs, band='r')
     
     # pl.plot(zs, MM)
     # pl.xlabel('z')
     # pl.ylabel('M')
     # pl.show()

     pl.clf()
     pl.plot(zs, mm)
     pl.xlabel('z')
     pl.ylabel('r')
     pl.show()
