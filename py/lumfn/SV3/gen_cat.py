import os
import sys
import time
import fitsio
import pylab             as     pl 
import numpy             as     np
import astropy.io.fits   as     fits

from   astropy.table     import Table, join
from   desitarget.sv3    import sv3_targetmask
from   matplotlib.pyplot import figure
from   scipy.optimize    import curve_fit

sys.path.append('/global/homes/m/mjwilson/desi/BGS/lumfn/bin')
sys.path.append('/global/homes/m/mjwilson/desi/BGS/lumfn/py/')
sys.path.append('/global/homes/m/mjwilson/desi/BGS/lumfn/py/lumfn/')

sys.path.append('/global/homes/m/mjwilson/desi/LSS/bin/')
sys.path.append('/global/homes/m/mjwilson/desi/LSS/py/')

from   distances         import comoving_distance, comoving_volume
from   ajs_kcorr         import ajs_kcorr
from   abs_mag           import abs_mag
from   vmax              import zmax, vmax
from   LSS.SV3.cattools  import tile2rosette  

def zsuccess(rfiber, a=2.68, b=22.113):
    return 1. /  (1. + np.exp(a * (rfiber - b)))

odir             = os.environ['CSCRATCH'] + '/desi/BGS/lumfn/'

# Targeting info. cut to (enlarged) area around rosettes.
targ             = Table.read('/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/bright_targets.fits')

# Cut to only BGS bright of available targets.
bgs_bright       = targ[(sv3_targetmask.bgs_mask['BGS_BRIGHT'] & targ['SV3_BGS_TARGET']) != 0]

# left outer join of reachable targets and spectro.
# cut to BGS Bright. 
bright_merge     = Table.read('/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/daily/datcomb_bright_tarspecwdup_Alltiles.fits')
bright_merge     = bright_merge[(sv3_targetmask.bgs_mask['BGS_BRIGHT'] & bright_merge['SV3_BGS_TARGET']) != 0]

# Limit to spectro. observations on a working fiber.
# Limit to first observation.  Subsequent observations reassigned on the basis of a bad first redshift.  Check: is 3000 true only of dark time?  Doubtful. 
bright_merge_obs                    = bright_merge[(bright_merge['ZWARN'].data != 999999) & (bright_merge['FIBERSTATUS'] == 0) & (bright_merge['PRIORITY'] > 3000.)]

# Add fluxes & colors. 
bright_merge_obs                    = join(bright_merge_obs, bgs_bright['FIBERFLUX_R', 'FLUX_G', 'FLUX_R', 'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'TARGETID'], join_type='left', keys='TARGETID')

bright_merge_obs['GMAG']            = 22.5 - 2.5 * np.log10(bright_merge_obs['FLUX_G']) 
bright_merge_obs['GMAG_DRED']       = 22.5 - 2.5 * np.log10(bright_merge_obs['FLUX_G'] / bright_merge_obs['MW_TRANSMISSION_G']) 

bright_merge_obs['RMAG']            = 22.5 - 2.5 * np.log10(bright_merge_obs['FLUX_R']) 
bright_merge_obs['RMAG_DRED']       = 22.5 - 2.5 * np.log10(bright_merge_obs['FLUX_R'] / bright_merge_obs['MW_TRANSMISSION_R']) 

bright_merge_obs['FIBER_RMAG']      = 22.5 - 2.5 * np.log10(bright_merge_obs['FIBERFLUX_R']) 
bright_merge_obs['FIBER_RMAG_DRED'] = 22.5 - 2.5 * np.log10(bright_merge_obs['FIBERFLUX_R'] / bright_merge_obs['MW_TRANSMISSION_R'])

bright_merge_obs['GMR']             = bright_merge_obs['GMAG'] - bright_merge_obs['RMAG']
bright_merge_obs['GMR_DRED']        = bright_merge_obs['GMAG_DRED'] - bright_merge_obs['RMAG_DRED']

bright_merge_obs['BGS_Z_SUCCESS']   = (bright_merge_obs['ZWARN'] == 0) & (bright_merge_obs['DELTACHI2'] > 40.)
bright_merge_obs['BGS_Z_WEIGHT']    = 1. / zsuccess(bright_merge_obs['FIBER_RMAG'])

# 300 km/s lower limit for stars; upper limit due to extrapolation of k correction. 
bright_merge_cat                    = bright_merge_obs[(bright_merge_obs['Z'] >= 0.001) & (bright_merge_obs['Z'] <= 0.5)]
bright_merge_cat.write('{}bright_sv3.fits'.format(odir), format='fits', overwrite=True)

derived     = []

start       = time.time()

kcorrector  = ajs_kcorr()

for ii, row in enumerate(bright_merge_cat):
    tid     = row['TARGETID']

    ros     = tile2rosette(row['ZTILEID'])
    
    rmag    = row['RMAG_DRED']
    gmr     = row['GMR_DRED']
    zz      = row['Z']

    wght    = row['BGS_Z_WEIGHT']
    
    vol     = comoving_volume(0.001, zz, fsky=1.0)
    
    Mrh     = abs_mag(kcorrector, rmag, gmr, zz).item()
    
    maxz    = zmax(kcorrector, 19.5, Mrh, gmr, zz)
    
    maxv    = vmax(kcorrector, 19.5, Mrh, gmr, zz, min_z=0.001, fsky=1.0, max_z=maxz)        
    
    vonvmax = vol / maxv
    
    derived.append([tid, ros, wght, vol, Mrh, maxz, maxv, vonvmax])

    if (ii % 100) == 0:
        runtime = (time.time()-start) / 60.
        
        percentage_complete = 100. * ii / len(bright_merge_cat)
        
        print('{:.2f} complete after {:.2f} minutes.'.format(percentage_complete, runtime))

        if runtime > .5:
            break

derived = Table(np.array(derived), names=['TARGETID', 'ROSETTE', 'BGSZWEIGHT', 'VOLUME', 'MRH', 'ZMAX', 'VMAX', 'VONVMAX'])

derived.pprint(max_width=-1)

derived.write('{}bright_sv3_derived.fits'.format(odir), format='fits', overwrite=True)

print('Finished writing to' + ' {}bright_sv3_derived.fits'.format(odir))
