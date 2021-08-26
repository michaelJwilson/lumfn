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
from   zsuccess          import zsuccess
from   asuccess          import asuccess

sys.path.append('/global/homes/m/mjwilson/desi/BGS/lumfn/bin')
sys.path.append('/global/homes/m/mjwilson/desi/BGS/lumfn/py/')
sys.path.append('/global/homes/m/mjwilson/desi/BGS/lumfn/py/lumfn/')

sys.path.append('/global/homes/m/mjwilson/desi/LSS/bin/')
sys.path.append('/global/homes/m/mjwilson/desi/LSS/py/')

from   distances         import comoving_distance, comoving_volume
from   ajs_kcorr         import ajs_kcorr
from   abs_mag           import abs_mag
from   vmax              import zmax, vmax
from   ref_gmr           import one_reference_gmr
from   LSS.SV3.cattools  import tile2rosette  
from   params            import params
from   define_sample     import define_sample
from   zmin              import zmin
from   sv3_params        import sv3_params

version          = 0.3
todisk           = True
dryrun           = False
runtime_lim      = 0.1 # only if dryrun. 
odir             = os.environ['CSCRATCH'] + '/desi/BGS/lumfn/'

print(odir)

# Targeting info. cut to (enlarged) area around rosettes.
targ             = Table.read('/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/bright_targets.fits')

# Cut to only BGS bright of available targets.
bgs_bright       = targ[(sv3_targetmask.bgs_mask['BGS_BRIGHT'] & targ['SV3_BGS_TARGET']) != 0]

# left outer join of reachable targets and spectro.
# cut to BGS Bright. 
bright_merge     = Table.read('/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/daily/datcomb_bright_tarspecwdup_Alltiles.fits')
bright_merge     = bright_merge[(sv3_targetmask.bgs_mask['BGS_BRIGHT'] & bright_merge['SV3_BGS_TARGET']) != 0]

# Add fluxes & colors.
bright_merge                        = join(bright_merge, bgs_bright['FIBERFLUX_R', 'FLUX_G', 'FLUX_R', 'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'TARGETID', 'PHOTSYS', 'REF_CAT'], join_type='left', keys='TARGETID')

bright_merge['GMAG']                = 22.5 - 2.5 * np.log10(bright_merge['FLUX_G'])
bright_merge['GMAG_DRED']           = 22.5 - 2.5 * np.log10(bright_merge['FLUX_G'] / bright_merge['MW_TRANSMISSION_G'])

bright_merge['RMAG']                = 22.5 - 2.5 * np.log10(bright_merge['FLUX_R'])
bright_merge['RMAG_DRED']           = 22.5 - 2.5 * np.log10(bright_merge['FLUX_R'] / bright_merge['MW_TRANSMISSION_R'])

bright_merge['FIBER_RMAG']          = 22.5 - 2.5 * np.log10(bright_merge['FIBERFLUX_R'])
bright_merge['FIBER_RMAG_DRED']     = 22.5 - 2.5 * np.log10(bright_merge['FIBERFLUX_R'] / bright_merge['MW_TRANSMISSION_R'])

bright_merge['GMR']                 = bright_merge['GMAG'] - bright_merge['RMAG']
bright_merge['GMR_DRED']            = bright_merge['GMAG_DRED'] - bright_merge['RMAG_DRED']

# Assignment (to a working fiber) success.  
bright_merge['BGS_A_SUCCESS']       = (bright_merge['ZWARN'].data != 999999) & (bright_merge['FIBERSTATUS'] == 0) 

bright_merge.pprint()

if todisk:
    bright_merge.write('{}bright_reachable_sv3_v{:.1f}.fits'.format(odir, version), format='fits', overwrite=True)

bright_merge['BGS_A_WEIGHT']          = 1. / asuccess(bright_merge['RMAG_DRED'])

# Limit to spectro. observations on a working fiber.
# Limit to first observation.  Subsequent observations reassigned on the basis of a bad first redshift.  Check: is 3000 true only of dark time?  Doubtful. 
# 300 km/s lower limit for stars; upper limit due to extrapolation of k correction.
bright_merge_obs                      = bright_merge[bright_merge['BGS_A_SUCCESS'] & (bright_merge['PRIORITY'] > 3000.)]

# (bright_merge_obs['Z'] >= 0.001) & (bright_merge_obs['Z'] <= 0.55)
# 
bright_merge_obs['BGS_Z_SUCCESS']     = (bright_merge_obs['ZWARN'] == 0) & (bright_merge_obs['DELTACHI2'] > 40.)
# bright_merge_obs['BGS_Z_SUCCESS']  &= (bright_merge_obs['ZTILEID'] > -1)
bright_merge_obs['BGS_Z_WEIGHT']      = 1. / zsuccess(bright_merge_obs['FIBER_RMAG'])

bright_merge_obs.pprint()

if todisk:
    bright_merge_obs.write('{}bright_sv3_v{:.1f}.fits'.format(odir, version), format='fits', overwrite=True)

## 
derived      = []

start        = time.time()

kcorrector   = ajs_kcorr()

fails        = []
tids         = []

for ii, row in enumerate(bright_merge_obs):
    tid      = row['TARGETID']
    tids.append(tid)
    
    ros      = tile2rosette(row['ZTILEID'])
    
    rmag     = row['RMAG_DRED']
    gmr      = row['GMR_DRED']
    zz       = row['Z']

    #  See dedicated fsky calc. notebook. 
    vol      = comoving_volume(zmin(params['vmin']), zz, fsky=sv3_params['fsky'])
    
    zwght    = row['BGS_Z_WEIGHT']
    awght    = row['BGS_A_WEIGHT']

    zsuccess = row['BGS_Z_SUCCESS']

    insample = define_sample(row)
    
    void     = [tid, ros, zsuccess, False, zwght, awght, -99., -99., -99., -99., -99., -99., -99.]

    if zsuccess & insample:
        try:            
            Mrh     = abs_mag(kcorrector, rmag, gmr, zz).item()

            org_gmr = one_reference_gmr(kcorrector, gmr, zz, zref=kcorrector.z0, ecorr=False)
            
            ref_gmr = one_reference_gmr(kcorrector, gmr, zz, zref=params['ref_z'])

            maxz    = zmax(kcorrector, params['rlim'], Mrh, gmr, zz, ref_gmr=ref_gmr)
    
            maxv    = vmax(kcorrector, params['rlim'], Mrh, gmr, zz, min_z=zmin(params['vmin']), fsky=sv3_params['fsky'], max_z=maxz)        
    
            vonvmax = (vol / maxv)
            
            derived.append([tid, ros, zsuccess, insample, zwght, awght, vol, Mrh, org_gmr, ref_gmr, maxz, 1. / maxv, vonvmax])

        except Exception as E:
            print('----  Exception  ----')
            print(E)
            print('----')
            
            fails.append(tid)
            derived.append(void)
    else:
        derived.append(void)
        
    if (ii % 100) == 0:
        runtime = (time.time()-start) / 60.
        
        percentage_complete = 100. * ii / len(bright_merge_obs)
        
        if dryrun & (runtime > runtime_lim):
            break
        
        print('{:.2f} complete after {:.2f} minutes.'.format(percentage_complete, runtime))

fails   = np.array(fails)

if todisk:
    np.savetxt('{}bright_sv3_derivedfails_v{:.1f}.txt'.format(odir, version), fails, fmt='%d')
        
derived = Table(np.array(derived), names=['TARGETID', 'ROSETTE', 'BGS_Z_SUCCESS', 'INSAMPLE', 'BGS_Z_WEIGHT', 'BGS_A_WEIGHT', 'VOLUME', 'MRH', 'REF_GMR0P1', 'REF_GMR0P0', 'ZMAX', 'IVMAX', 'VONVMAX'])
derived['TARGETID'] = np.array(tids, dtype=np.int64)

print(np.mean(derived['INSAMPLE']))

# Exclude those from sample that had more than one shot at a redshift. 
exclude = np.array([x == b'BGS|MORE_ZWARN' for x in bright_merge_obs['TARGET_STATE'].data])
exclude = bright_merge_obs['TARGETID'].data[exclude]
exclude = np.isin(derived['TARGETID'].data, exclude)
derived['INSAMPLE'].data[exclude] = 0.0

print('Excluded as repeated-attempt redshifts: {:d}'.format(np.count_nonzero(exclude)))

derived.pprint(max_width=-1)

if todisk:
    derived.write('{}bright_sv3_derived_v{:.1f}.fits'.format(odir, version), format='fits', overwrite=True)

print('Finished writing to' + ' {}bright_sv3_derived_v{:.1f}.fits'.format(odir, version))
