import numpy as np
import matplotlib.pyplot as plt

from   pkg_resources     import resource_filename
from   astropy.table     import Table
from   scipy.interpolate import interp1d
from   scipy.stats       import linregress
from   tmr_ecorr         import tmr_ecorr
from   params            import params

# See: https://arxiv.org/pdf/1409.4681.pdf
#      https://arxiv.org/pdf/1701.06581.pdf
# 
class ajs_kcorr():
    def __init__(self):
        self.z0      = 0.1
        self.zmin    = 1.e-16 
        
        self.raw_dir = resource_filename('lumfn', 'data/')
        
        self.names   = ['gmr_min', 'gmr_max', 'A0', 'A1', 'A2', 'A3', 'A4', 'gmr_med']
        
        self.rawg    = Table(np.loadtxt(self.raw_dir + '/ajs_kcorr_gband_z01.dat'), names=self.names)
        self.rawr    = Table(np.loadtxt(self.raw_dir + '/ajs_kcorr_rband_z01.dat'), names=self.names)
        
        # Note: gmr color bounds are common to both. 
        self.clims   = self.rawg['gmr_min', 'gmr_max', 'gmr_med']

        # Keep only coeff. tables.                                                                                                                                                            
        for nm in ['gmr_min', 'gmr_max', 'gmr_med']:
            del self.rawg[nm]
            del self.rawr[nm]
        
        # For An * z^n dot product. 
        self.base  = 4 - np.arange(0, 5, 1)

        self.Ans    = {'g': {'raw': self.rawg}, 'r': {'raw': self.rawr}}
        
        for band in ['g', 'r']:
            for nm in ['A0', 'A1', 'A2', 'A3', 'A4']:
                # Interpolators in color.  Note extrapolation call. 
                self.Ans[band][nm] = interp1d(self.clims['gmr_med'], self.Ans[band]['raw'][nm], kind='linear', fill_value='extrapolate')
                
            # Fall back for A4. 
            self.Ans[band]['A4'] = lambda x: self.Ans[band]['raw']['A4'][0]

        self.prep_extrap()
                    
        self.eval    = np.vectorize(self.__eval)
        
    def _eval(self, ref_gmr, zz, band):
        zz           = np.atleast_1d(np.array(zz, copy=True)) 
        
        # Clip passed observed color. 
        ref_gmr      = np.clip(ref_gmr, np.min(self.clims['gmr_med']), np.max(self.clims['gmr_med']))
        
        # Get coefficients at this color. 
        aa           = np.array([self.Ans[band][x](ref_gmr) for x in ['A0', 'A1', 'A2', 'A3', 'A4']])

        idx          = (zz <= self.z0)
        
        zz          -= self.z0        
        zz[idx]      = -zz[idx]
        
        sgns         = np.ones_like(zz)[:,None] * np.array([(-1) ** n for n in self.base])[None,:]
        sgns[~idx,:] = 1.0

        zz           = np.exp(np.log(self.zmin + zz)[:,None] * self.base[None,:])
        zz          *= sgns
        
        res          = aa * zz        
        res          = np.sum(res, axis=1)
        
        return res

    def prep_extrap(self):
        zs = [0.48, 0.50]

        for band in ['g', 'r']:
            slopes = []
            intercepts = []
            
            for gmr in self.clims['gmr_med']:
                ys = self._eval(gmr, zs, band)
            
                slope, intercept, _, _, _ = linregress(zs,ys)

                slopes.append(slope)
                intercepts.append(intercept)

            self.Ans[band]['L0'] = interp1d(self.clims['gmr_med'], np.array(intercepts), kind='linear', fill_value='extrapolate')
            self.Ans[band]['L1'] = interp1d(self.clims['gmr_med'], np.array(slopes),     kind='linear', fill_value='extrapolate')

    def ref_eval(self, ref_gmr, zz, band):
        zz            = np.atleast_1d(np.array(zz, copy=True))
        
        res           = self._eval(ref_gmr, zz, band)
        res[zz > 0.5] = self.Ans[band]['L0'](ref_gmr) + self.Ans[band]['L1'](ref_gmr) * zz[zz > 0.5]

        return res

    def __eval(self, ref_gmr, zz, band, ref_z=params['ref_z'], res=None, ecorr=True):
        # Applies reference z shift.
        if res == None:
            res       = self.ref_eval(ref_gmr, zz, band)

        if ref_z != self.z0:
            shift     = self.ref_eval(ref_gmr, ref_z, band) + 2.5 * np.log10(1. + ref_z)
            res      -= shift

        if ecorr & (ref_z == 0.0):
            tt        = 'blue' if (ref_gmr <= params['rf_gmr_redblue']) else 'red'
            res      += tmr_ecorr(zz, tt=tt, zref=ref_z, band=band)

        elif ecorr & (ref_z != 0.0):
            raise ValueError('E-correction only defined for a reference z of 0.0;')

        else:
            pass

        return  res
    
if __name__ == '__main__':
    import pylab as pl

    
    x         = ajs_kcorr()

    ref_gmrs  = [0.130634, 0.298124, 0.443336, 0.603434, 0.784644, 0.933226, 1.0673]
    zs        = np.arange(0.01,0.801,0.01)

    fig, axes = plt.subplots(1,2, figsize=(15,5))

    colors    = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for ref_gmr, color in zip(ref_gmrs, colors):
        # compare _eval to ref_eval, compares +- extrapolation.   
        gks = x._eval(ref_gmr, zs, 'g')
        rks = x._eval(ref_gmr, zs, 'r')
        
        axes[0].plot(zs, rks, label='', alpha=0.25, c=color)
        axes[1].plot(zs, gks, label='', alpha=0.25, c=color)

        gks = x.ref_eval(ref_gmr, zs, 'g')
        rks = x.ref_eval(ref_gmr, zs, 'r')

        axes[0].plot(zs, rks, '--', c=color, alpha=0.25)
        axes[1].plot(zs, gks, '--', c=color, alpha=0.25)

        gks = x.eval(ref_gmr, zs, 'g', ref_z=0.00, ecorr=False)
        rks = x.eval(ref_gmr, zs, 'r', ref_z=0.00, ecorr=False)

        axes[0].plot(zs, rks, '--', c=color)
        axes[1].plot(zs, gks, '--', c=color)

        gks = x.eval(ref_gmr, zs, 'g', ref_z=0.00, ecorr=True)
        rks = x.eval(ref_gmr, zs, 'r', ref_z=0.00, ecorr=True)

        axes[0].plot(zs, rks, c=color, label='ref. (g-r)={:.2f}'.format(ref_gmr))
        axes[1].plot(zs, gks, c=color)

    axes[0].set_ylabel(r"$^{0.1}K_r(z)$")
    axes[1].set_ylabel(r"$^{0.1}K_g(z)$")
    
    for ax in axes:
        ax.set_xlabel(r"$z$")
        ax.set_xlim(-0.01,0.6)
        ax.set_ylim(-0.4,1)
        ax.legend(loc=2, frameon=False)

    pl.show()
    
