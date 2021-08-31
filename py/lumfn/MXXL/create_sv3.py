import sys
import h5py
import numpy as np

from   astropy.table import Table

sys.path.append('/global/homes/m/mjwilson/desi/BGS/lumfn/py/lumfn/MXXL/')

from   rotate_mock import rotate_mock

# /global/project/projectdirs/desi/mocks/bgs/MXXL/one_percent/README
fpath = '/project/projectdirs/desi/mocks/bgs/MXXL/full_sky/v0.0.4/BGS_r20.6.hdf5'

f     = h5py.File(fpath, mode='r')

ra    = f["Data/ra"][...]
dec   = f["Data/dec"][...]
z     = f["Data/z_obs"][...]
zcos  = f["Data/z_cos"][...]
r     = f["Data/app_mag"][...]
gmr   = f["Data/g_r"][...]
tt    = f["Data/galaxy_type"][...]
hmass = f["Data/halo_mass"][...]

# 
fpath = '/global/project/projectdirs/desi/mocks/bgs/MXXL/one_percent/one_percent_v2.hdf5'

f     = h5py.File(fpath, mode='r') 
nmock = f['nmock'][:]

isin  = (r <= 19.5) & (nmock > -1)

print(100. * np.mean(isin))

names = ['MOCKRA', 'MOCKDEC', 'Z', 'ZCOS', 'RMAG_DRED', 'REFGMR0P1', 'GTYPE', 'HMASS', 'NMOCK']
data  = Table(np.c_[ra[isin], dec[isin], z[isin], zcos[isin], r[isin], gmr[isin], tt[isin], hmass[isin], nmock[isin]], names=names) 
data['NMOCK'] = data['NMOCK'].data.astype(np.int)

data['RA']    = -99.
data['DEC']   = -99.


uns   = np.unique(data['NMOCK'].data)

for un in uns:
    isin        = data['NMOCK'] == un

    _ras, _decs = rotate_mock(data['MOCKRA'].data[isin], data['MOCKDEC'].data[isin], np.int(un), inverse=False, version=2)

    data['RA'][isin]  = _ras
    data['DEC'][isin] = _decs

root  = "/global/cscratch1/sd/mjwilson/desi/BGS/lumfn/MXXL/"
fpath = root + "galaxy_catalogue_sv3s.fits"

data.write(fpath, format='fits', overwrite=True)

print('Done.')
