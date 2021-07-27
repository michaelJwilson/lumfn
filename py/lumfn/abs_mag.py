import scipy
import numpy     as     np
import pylab     as     pl 

from   distances import comoving_distance, dist_mod
from   tmr_kcorr import tmr_kcorr
from   tmr_ecorr import	tmr_ecorr


def abs_mag(kcorr, rpetro, obs_gmr, zz):
     chi = comoving_distance(zz)
     mu  = dist_mod(chi)

     kk  = kcorr.eval(obs_gmr, zz)

     tt  = 'gray'  # Base on obs_gmr.
     EE  = tmr_ecorr(tt, zz)
     
     #  M - 5log10(h)
     return  rpetro - mu - kk - EE

def app_mag(kcorr, Mrh, zz):
     # Mrh \equiv Mr - 5log10(h)
     chi = comoving_distance(zz)
     mu  = dist_mod(chi)

     # TODO: predict obs_gmr given varying z. 

     '''
     kk  = kcorr.eval(obs_gmr, zz)

     tt  = 'gray'  # Base on obs_gmr.                                                                                                                                                                                   
     EE  = tmr_ecorr(tt, zz)
     '''
     return Mrh + mu

#  Cases to catch. 

#  Stellar, negative redshift. 
#  18.045475 0.77989006 -0.0001094191322428999 -0.14781370053573228 [nan]

#  Half a magnitude redder than Smith et al.; 
#  19.270914 0.64866066 1.4671619464660783 1933467373610.0474 [-61.1826436]
     
if __name__ == '__main__':
     x   = tmr_kcorr()
     zs  = np.arange(0.1, 0.6, 0.05)
     
     MM  = abs_mag(x, 19.8, 0.708, zs)
     mm  = app_mag(x, -28., zs)
     
     # pl.plot(zs, MM)
     # pl.xlabel('z')
     # pl.ylabel('M')
     # pl.show()

     pl.clf()
     pl.plot(zs, mm)
     pl.xlabel('z')
     pl.ylabel('r')
     pl.show()
