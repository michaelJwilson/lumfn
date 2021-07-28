import numpy as np
import matplotlib.pyplot as plt

from   pkg_resources     import resource_filename
from   tmr_ecorr         import tmr_ecorr
from   astropy.table     import Table
from   scipy.interpolate import interp1d

# See: https://arxiv.org/pdf/1409.4681.pdf
#      https://arxiv.org/pdf/1701.06581.pdf
# 
class ajs_kcorr():
    def __init__(self):
        self.z0      = 0.1
        self.raw_dir = resource_filename('lumfn', 'data/')
        
        self.names = ['gmr_min', 'gmr_max', 'A0', 'A1', 'A2', 'A3', 'A4', 'gmr_med']
        
        self.rawg  = Table(np.loadtxt(self.raw_dir + '/ajs_kcorr_gband_z01.dat'), names=self.names)
        self.rawr  = Table(np.loadtxt(self.raw_dir + '/ajs_kcorr_rband_z01.dat'), names=self.names)
        
        # Note: gmr color bounds are common to both. 
        self.clims = self.rawg['gmr_min', 'gmr_max', 'gmr_med']

        # Keep only coeff. tables.                                                                                                                                                            
        for nm in ['gmr_min', 'gmr_max', 'gmr_med']:
            del self.rawg[nm]
            del self.rawr[nm]
        
        # For An * z^n dot product. 
        self.base  = 4 - np.arange(0, 5, 1)
        
        # Seed interpolators. 
        self.gAns  = {}
        self.rAns  = {}

        for nm in ['A0', 'A1', 'A2', 'A3', 'A4']:
            # Interpolators in color.  Note extrapolation call. 
            self.gAns[nm] = interp1d(self.clims['gmr_med'], self.rawg[nm], kind='linear', fill_value='extrapolate')
            self.rAns[nm] = interp1d(self.clims['gmr_med'], self.rawr[nm], kind='linear', fill_value='extrapolate')

        # Fall back for A4. 
        self.gAns['A4'] = lambda x: self.rawg['A4'][0]
        self.rAns['A4'] = lambda x: self.rawr['A4'][0]

        self.Ans = {'g': self.gAns, 'r': self.rAns}
        
    def eval(self, obs_gmr, zz, band):
        zz      = np.atleast_1d(zz) 
        
        # Clip passed observed color. 
        obs_gmr = np.clip(obs_gmr, np.min(self.clims['gmr_med']), np.max(self.clims['gmr_med']))
        
        # Get coefficients at this color. 
        aa      = np.array([self.Ans[band][x](obs_gmr) for x in ['A0', 'A1', 'A2', 'A3', 'A4']])

        # Fails below z0. 
        zz      = np.exp(np.log(zz - self.z0)[:,None] * self.base[None,:])
        
        res     = aa * zz        
        res     = np.sum(res, axis=1)
        
        return res
    
 
if __name__ == '__main__':
    import pylab as pl
        
    x    = ajs_kcorr()

    gmrs = [0.130634, 0.298124, 0.443336, 0.603434, 0.784644, 0.933226, 1.0673]
    zs   = np.arange(0.11,0.601,0.01)

    fig, axes = plt.subplots(1,2, figsize=(15,5))
    
    for gmr in gmrs:
        gks = x.eval(gmr, zs, 'g')
        rks = x.eval(gmr, zs, 'r')
        
        axes[0].plot(zs, rks, label=r"$(g-r)=%.3f$" % gmr)
        axes[1].plot(zs, gks, label=r"$(g-r)=%.3f$" % gmr)

    axes[0].set_ylabel(r"$^{0.1}K_r(z)$")
    axes[1].set_ylabel(r"$^{0.1}K_g(z)$")
    
    for ax in axes:
        ax.set_xlabel(r"$z$")
        ax.set_xlim(-0.01,0.6)
        ax.set_ylim(-0.4,1)
        ax.legend(loc=2, frameon=False)

    pl.show()
    
