import numpy as np

from   params import params

def define_sample(bright_merge_obs, vmin=params['vmin'], printit=False):
    isin = (bright_merge_obs['BGS_Z_SUCCESS']) & (bright_merge_obs['SPECTYPE'] != 'STAR')

    # redrock misidentifies stars as galaxies, but at v < 200 km/s.
    isin = isin & (bright_merge_obs['Z'] >= (vmin / 2.9979e5))

    # Color corrections well-defined to 0.5, extrapolated otherwise. 
    isin = isin & (bright_merge_obs['Z'] <= params['zmax'])

    if printit:
        print('Selecting {:.3f}% of sample'.format(100. * np.mean(isin)))

    return  isin
