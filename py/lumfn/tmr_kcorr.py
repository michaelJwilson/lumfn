import numpy as np

from   pkg_resources import resource_filename
from   tmr_ecorr     import tmr_ecorr

# https://arxiv.org/pdf/1409.4681.pdf
class tmr_kcorr():
    def __init__(self):
        self.raw_dir = resource_filename('lumfn', 'data/')
        self.raw = np.loadtxt(self.raw_dir + '/tmr_kcorr.txt')
        self.base = 4 - np.arange(0, 5, 1)
        
    def eval(self, obs_gmr, zz):
        zz  = np.atleast_1d(zz) 
        
        idx = np.digitize(obs_gmr, self.raw[:,0], right=True)
        aa  = self.raw[idx, 1:]
        zz  = np.exp(np.log(zz)[:,None] * self.base[None,:])

        res = aa * zz        
        res = np.sum(res, axis=1)
        
        return res
        
if __name__ == '__main__':
    x = tmr_kcorr()

    x.eval(0.157, 0.1)
