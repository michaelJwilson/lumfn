import numpy as np

from   scipy.interpolate import interp1d
from   pkg_resources     import resource_filename


data_dir = resource_filename('lumfn', 'data/')
ecorr = np.loadtxt(data_dir + '/mxxl_ecorr.txt')

# [-23.82841, -2.7]
mxxl_ecorr = interp1d(ecorr[:,0], ecorr[:,1], kind='linear', copy=True, bounds_error=True)
