import numpy as np

from   scipy.interpolate import interp1d
from   pkg_resources     import resource_filename


data_dir = resource_filename('lumfn', 'data/')
limits   = np.loadtxt(data_dir + '/kmax.txt')

# [-23.82841, -2.7]
zmin     = interp1d(limits[:,0], limits[:,2], kind='linear', copy=True, bounds_error=True)

# [-22.22539, -2.6]
zmax     = interp1d(limits[:,1], limits[:,2], kind='linear', copy=True, bounds_error=True)

kmax     = interp1d(limits[:,2], limits[:,3], kind='linear', copy=True, bounds_error=True)

'''
data_dir = resource_filename('lumfn', 'data/')
limits = np.loadtxt(data_dir + '/kmax_tmr.txt')

# [-23.82841, -2.7]                                                                                                                                                                                                                        
tmr_zmin = interp1d(limits[:,0], limits[:,2], kind='linear', copy=True, bounds_error=True)

# [-22.22539, -2.6]
tmr_zmax = interp1d(limits[:,1], limits[:,2], kind='linear', copy=True, bounds_error=True)
'''
if __name__ == '__main__':
    import pylab as pl

    Ms = np.arange(-22.1, -18., 0.1)
    
    pl.plot(Ms, zmin(Ms), label='zmin')
    pl.plot(Ms, zmax(Ms), label='zmax')

    pl.show()
