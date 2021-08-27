import numpy        as     np

from   params       import params
from   zmin         import zmin  
from   GAMA.in_gama import in_gama


def define_sample(bright_merge_obs, vmin=params['vmin'], zmax=params['zmax'], printit=False):
    isin = (bright_merge_obs['BGS_Z_SUCCESS'])

    # redrock misidentifies stars as galaxies, but at v < 200 km/s.
    isin = isin & (bright_merge_obs['ZGAMA'] >= zmin(vmin))

    # Color corrections well-defined to 0.5, extrapolated otherwise. 
    isin = isin & (bright_merge_obs['ZGAMA'] <= zmax)

    isin = isin & (bright_merge_obs['RMAG_DRED'] <= params['rlim'])

    isin = isin & in_gama(bright_merge_obs['RA'], bright_merge_obs['DEC'])
    
    if printit:
        print('Selecting {:.3f}% of sample'.format(100. * np.mean(isin)))

    return  isin
