import matplotlib.pyplot as plt
import numpy             as np

from   pkg_resources     import resource_filename
from   params            import params

# See: https://arxiv.org/pdf/1409.4681.pdf
#      https://arxiv.org/pdf/1701.06581.pdf
# 
class tmr_kcorr():
    def __init__(self):
        self.z0 = 0.0
        
        self.raw_dir = resource_filename('lumfn', 'data/')
        self.raw = np.loadtxt(self.raw_dir + '/tmr_kcorr.txt')
        self.base = 4 - np.arange(0, 5, 1)
        
    def ref_eval(self, ref_gmr, zz, band='r', ref_z=params['ref_z']):
        assert ref_z == 0.0, "Only z=0.0 reference is supported."
        
        zz  = np.atleast_1d(np.array(zz, copy=True))
        
        idx = np.digitize(ref_gmr, self.raw[:,0], right=True)
        idx = np.minimum(idx, len(self.raw) - 1)
        
        aa  = self.raw[idx, 1:]
        zz  = np.exp(np.log(zz)[:,None] * self.base[None,:])
        
        res = aa * zz        
        res = np.sum(res, axis=1)
        
        return res
        
if __name__ == '__main__':
    import pylab as pl


    x         = tmr_kcorr()

    ref_gmrs  = [0.158, 0.298, 0.419, 0.553, 0.708, 0.796, 0.960]
    zs        = np.arange(0.01,0.801,0.01)

    
    fig, axes = plt.subplots(1,2, figsize=(15,5))

    colors    = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for ref_gmr, color in zip(ref_gmrs, colors):
        # compare _eval to ref_eval, compares +- extrapolation.                                                                                                                                                             
        # gks = x.ref_eval(ref_gmr, zs, 'g')
        rks = x.ref_eval(ref_gmr, zs, 'r')

        axes[0].plot(zs, rks, '-', c=color, alpha=0.75, label='ref. $(g-r)$: {:.3f}'.format(ref_gmr))
        # axes[1].plot(zs, gks, '--', c=color, alpha=0.25)

    axes[0].set_ylabel(r"$^{0.1}K_r(z)$")
    axes[1].set_ylabel(r"$^{0.1}K_g(z)$")

    for ax in axes:
        ax.set_xlabel(r"$z$")
        ax.set_xlim(-0.01,0.5)
        ax.set_ylim(-0.4,1.1)

        ax.legend(loc=2, frameon=False)

    pl.show()




    
