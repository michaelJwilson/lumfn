import numpy as np

from zlimits import zmin

def set_ddps(joined, Mlo=-22., Mup=-16.):
    dM   = 0.25
    lims = np.arange(Mlo, Mup, dM)

    joined['DDPS']    = np.digitize(joined['MRH'].data, bins=lims)
    joined['DDPZMAX'] = -1.
    
    for idx in np.arange(0, len(lims), 1):
        zlim = zmin(lims[idx])

        isin = (joined['DDPS'].data == idx) & (joined['Z'].data > zlim)

        joined['DDPZMAX'].data[(joined['DDPS'].data == idx)] = zlim
        
        joined['DDPS'].data[isin] = -1
        joined['DDPZMAX'].data[isin] = -1
        
    joined['DDPS'][joined['MRH'] > Mup] = -1
    joined['DDPS'][joined['MRH'] < Mlo] = -1

    joined['DDPZMAX'][joined['MRH'] > Mup] = -1
    joined['DDPZMAX'][joined['MRH'] < Mlo] = -1

    return lims
