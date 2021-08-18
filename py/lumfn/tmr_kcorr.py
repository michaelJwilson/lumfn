import numpy as np

from   pkg_resources import resource_filename
from   tmr_ecorr     import tmr_ecorr

# See: https://arxiv.org/pdf/1409.4681.pdf
#      https://arxiv.org/pdf/1701.06581.pdf
# 
class tmr_kcorr():
    def __init__(self):
        self.z0 = 0.0
        
        self.raw_dir = resource_filename('lumfn', 'data/')
        self.raw = np.loadtxt(self.raw_dir + '/tmr_kcorr.txt')
        self.base = 4 - np.arange(0, 5, 1)
        
    def eval(self, obs_gmr, zz, band='r'):
        zz  = np.atleast_1d(zz) 
        
        idx = np.digitize(obs_gmr, self.raw[:,0], right=True)
        idx = np.minimum(idx, len(self.raw) - 1)
        
        aa  = self.raw[idx, 1:]
        zz  = np.exp(np.log(zz)[:,None] * self.base[None,:])

        res = aa * zz        
        res = np.sum(res, axis=1)
        
        return res
        
if __name__ == '__main__':
    x = tmr_kcorr()

    x.eval(0.157, 0.1)
