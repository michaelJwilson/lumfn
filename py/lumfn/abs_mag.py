import scipy
import numpy     as     np
import pylab     as     pl 

from   distances import comoving_distance, dist_mod

from   ajs_kcorr  import ajs_kcorr
from   mxxl_ecorr import mxxl_ecorr
from   tmr_ecorr  import tmr_ecorr
from   ref_gmr    import ajs_reference_gmr, tmr_reference_gmr


def abs_mag(kcorr, rpetro, obs_gmr, zz, band='r', ref_gmr=None, tmr=False):
     chi = comoving_distance(zz)
     mu  = dist_mod(zz)

     if ref_gmr == None:
          ref_gmr = ajs_reference_gmr(kcorr, obs_gmr, zz)
     
     # Note: tmr references to z=0.0; ajs to z=0.1;
     kk  = kcorr.eval(ref_gmr, zz, band=band)

     if tmr:
          kk = kcorr.shift_refz(kk, ref_gmr, ref_z=0.0, band=band)          

          tmr_ref_gmr = tmr_reference_gmr(kcorr, ref_gmr)
               
          kk += tmr_ecorr(zz, tmr_ref_gmr)

     else:
          kk += mxxl_ecorr(zz)
          
     #  Returns:  M - 5log10(h)
     #  See https://arxiv.org/pdf/1409.4681.pdf
     return rpetro - mu - kk

def app_mag(kcorr, Mh, obs_gmr, zz, band='r', ref_gmr=None, tmr=False):     
     # Mh \equiv Mr - 5log10(h)
     chi = comoving_distance(zz)
     mu  = dist_mod(zz)

     if ref_gmr == None:
          ref_gmr = ajs_reference_gmr(kcorr, obs_gmr, zz)

     kk  = kcorr.eval(ref_gmr, zz, band=band)

     if tmr:
          kk = kcorr.shift_refz(kk, ref_gmr, ref_z=0.0, band=band)

          tmr_ref_gmr = tmr_reference_gmr(kcorr, ref_gmr)
          
          kk += tmr_ecorr(zz, tmr_ref_gmr)

     else:
          kk += mxxl_ecorr(zz)
          
     return Mh + mu + kk


if __name__ == '__main__':
     x       = ajs_kcorr()  # [ajs_kcorr, tmr_kcorr]

     zz      =   0.2
     
     rp      =  19.5
     Mh      = -21.0

     band    =    'r'
     
     obs_gmr = 1.000 

     ref_gmrs = [0.130634, 0.298124, 0.443336, 0.603434, 0.784644, 0.933226, 1.0673]

     zs = np.arange(0.01, 0.5, 0.005)
     
     for ref_gmr in [ref_gmrs[0], ref_gmrs[-1]]:
          MM = []
          MMp = []

          tmr_ref_gmr = tmr_reference_gmr(x, ref_gmr)[0]
          
          for zz in zs:
               '''
               chi     = comoving_distance(zz)
               mu      = dist_mod(zz)

               ref_gmr = ajs_reference_gmr(x, obs_gmr, zz)

               kk      = x.eval(ref_gmr, zz, band=band)
               kkE     = x.eval(ref_gmr, zz, band=band)
               '''

               MM.append(abs_mag(x, rp, obs_gmr, zz, band='r', tmr=False, ref_gmr=ref_gmr).item())
               MMp.append(abs_mag(x, rp, obs_gmr, zz, band='r', tmr=True,  ref_gmr=ref_gmr).item())
               
               # pl.plot(zz, mu + kk,  marker='.', lw=0.0, c='k')
               # pl.plot(zz, mu + kkE, marker='.', lw=0.0, c='c')
          
               # pl.plot(zz, mm, marker=',', lw=0.0)

          MM = np.array(MM)
          MMp = np.array(MMp)

          pl.plot(zs, MM,  label='TMR: 0 ({:.3f}, {:.3f})'.format(ref_gmr, tmr_ref_gmr), alpha=0.5)
          pl.plot(zs, MMp, label='TMR: 1 ({:.3f}, {:.3f})'.format(ref_gmr, tmr_ref_gmr))
          
     # pl.axvline(0.25, lw=0.25, c='k')
          
     # pl.plot(zs, MM)
     # pl.xlabel('z')
     # pl.ylabel('M')
     # pl.show()
     
     # pl.clf()
     # pl.plot(zs, mm)
     pl.xlabel('z')
     # pl.ylabel('$\mu + k&E$')
     # pl.ylabel('r')
     pl.legend()
     pl.show()
