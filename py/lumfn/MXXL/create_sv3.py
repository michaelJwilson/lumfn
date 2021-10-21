import sys
import h5py
import numpy as np

from   astropy.table import Table

sys.path.append('/global/homes/m/mjwilson/desi/BGS/lumfn/py/lumfn/MXXL/')

from   rotate_mock import rotate_mock
from desimodel.footprint import is_point_in_desi 

def tile2rosette(tile):
    if tile < 433:
        return (tile-1)//27
    else:
        if tile >= 433 and tile < 436:
            return 13
        if tile >= 436 and tile < 439:
            return 14
        if tile >= 439 and tile < 442:
            return 15
        if tile >= 442 and tile <=480:
            return (tile-442)//3
            
        if tile > 480:
            return tile//30    
    return 999999 #shouldn't be any more?

# /global/project/projectdirs/desi/mocks/bgs/MXXL/one_percent/README
# fpath = '/project/projectdirs/desi/mocks/bgs/MXXL/full_sky/v0.0.4/BGS_r20.6.hdf5'
fpath = '/project/projectdirs/desi/mocks/MXXL/v4.0/full_sky/galaxy_catalogue_r20.6.hdf5'

f     = h5py.File(fpath, mode='r')

ra    = f["Data/ra"][...]
dec   = f["Data/dec"][...]
zobs  = f["Data/z_obs"][...]
z     = f["Data/z_cos"][...]
r     = f["Data/app_mag"][...]
M     = f["Data/abs_mag"][...]
gmr   = f["Data/g_r"][...]
tt    = f["Data/galaxy_type"][...]
hmass = f["Data/halo_mass"][...]

# 
fpath = '/project/projectdirs/desi/mocks/MXXL/v4.0/one_percent/one_percent_v2.hdf5'

f     = h5py.File(fpath, mode='r') 
nmock = f['nmock'][:]

print(f.keys())

isin  = (r <= 19.5) & (nmock > -1)

print(len(ra), len(nmock), 100. * np.mean(isin))

names = ['MOCKRA', 'MOCKDEC', 'Z', 'ZOBS', 'MRH', 'RMAG_DRED', 'REFGMR0P1', 'GTYPE', 'HMASS', 'NMOCK']
data  = Table(np.c_[ra[isin], dec[isin], z[isin], zobs[isin], M[isin], r[isin], gmr[isin], tt[isin], hmass[isin], nmock[isin]], names=names) 
data['NMOCK'] = data['NMOCK'].data.astype(np.int)

data['RA']    = -99.
data['DEC']   = -99.

uns   = np.unique(data['NMOCK'].data)

for un in uns:
    isin        = data['NMOCK'] == un

    _ras, _decs = rotate_mock(data['MOCKRA'].data[isin], data['MOCKDEC'].data[isin], np.int(un), inverse=False, version=2)

    data['RA'][isin]  = _ras
    data['DEC'][isin] = _decs

## 210 tiles.
tiles            = Table.read('/global/cfs/cdirs/desi/spectro/redux/everest/tiles-everest.csv')
tiles['RA']      = tiles['TILERA'].data
tiles['DEC']     = tiles['TILEDEC'].data
tiles            = tiles[tiles['FAFLAVOR'].data == 'sv3bright']
tiles            = tiles[tiles['OBSSTATUS'] == 'obsend']
tiles['ROSETTE'] = np.array([tile2rosette(tileid) for tileid in tiles['TILEID'].data])

tiles.pprint()


isin, idx = is_point_in_desi(tiles, data['RA'].data, data['DEC'].data, return_tile_index=True)

data['TILEID']   = tiles['TILEID'].data[idx]
data['ROSETTE']  = tiles['ROSETTE'].data[idx]
data['TARGETID'] = np.arange(len(data), dtype=np.int64)

root  = "/global/cscratch1/sd/mjwilson/desi/BGS/lumfn/MXXL/"
fpath = root + "galaxy_catalogue_sv3s_v3.fits"

data.write(fpath, format='fits', overwrite=True)

print('Done {}.'.format(fpath))
