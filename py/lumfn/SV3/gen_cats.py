import os
import sys
import time
import fitsio
import pylab             as     pl 
import numpy             as     np
import astropy.io.fits   as     fits

from   astropy.table     import Table, join, vstack
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
from   rlim              import rlim

version          = 0.5
todisk           = True
dryrun           = False
runtime_lim      = 1.0 # only if dryrun. 
odir             = os.environ['CSCRATCH'] + '/desi/BGS/lumfn/'

print(version, odir)

# Targeting info. cut to (enlarged) area around rosettes.
targ             = Table.read('/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/bright_targets.fits')

# Cut to only BGS bright of available targets.
bgs_bright       = targ[(sv3_targetmask.bgs_mask['BGS_BRIGHT'] & targ['SV3_BGS_TARGET']) != 0]

# left outer join of reachable targets and spectro.
# cut to BGS Bright.
#
# fpath = '/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/daily/datcomb_bright_tarspecwdup_Alltiles.fits'
fpath = '/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/everest/LSScats/test/BGS_ANYAlltiles_full.dat.fits'

bright_merge     = Table.read(fpath)
bright_merge     = bright_merge[(sv3_targetmask.bgs_mask['BGS_BRIGHT'] & bright_merge['SV3_BGS_TARGET']) != 0]

#  Clustering cat:
# [('RA', '>f8'), ('DEC', '>f8'), ('TARGETID', '>i8'), ('Z', '>f8'), ('NTILE', '>i8'), ('TILES', 'S43'), ('rosette_number', '>f8'), ('WEIGHT', '>f8')]))
# /global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/LSScats/v0.1/BGS_BRIGHTAlltiles_clustering.dat.fits

# one shot only. 
bright_merge     = bright_merge[bright_merge['PRIORITY_INIT'] == 102100]

# Add targeting info. (fluxes & colors).
bright_merge     = join(bright_merge, bgs_bright['FIBERFLUX_R', 'FLUX_G', 'FLUX_R', 'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'TARGETID', 'PHOTSYS', 'REF_CAT', 'BRICK_OBJID', 'BRICKID'], join_type='left', keys='TARGETID')

# ----  Add GAMA info  ----.
# unique identifier is (release,brickid,objid); release assumed to be DR9 for both.
gama             = Table.read('/global/cfs/cdirs/desi/target/analysis/truth/dr9.0/south/matched/GAMA-DR3-SpecObj-match.fits')
gama['ZGAMA']    = gama['Z']

del gama['Z']
del gama['RA']
del gama['DEC']

igama               = Table.read('/global/cfs/cdirs/desi/target/analysis/truth/dr9.0/south/matched/ls-dr9.0-GAMA-DR3-SpecObj-match.fits')

gama['RELEASE']     = igama['RELEASE']
gama['BRICKID']     = igama['BRICKID']
gama['BRICK_OBJID'] = igama['OBJID']

ncheck = len(bright_merge)

bright_merge        = join(bright_merge, gama, join_type='left', keys=['BRICKID', 'BRICK_OBJID'])
bright_merge.pprint()

assert len(bright_merge) == ncheck

# ---- Add fluxes and colors. 
bright_merge['GMAG']                = 22.5 - 2.5 * np.log10(bright_merge['FLUX_G'])
bright_merge['GMAG_DRED']           = 22.5 - 2.5 * np.log10(bright_merge['FLUX_G'] / bright_merge['MW_TRANSMISSION_G'])

bright_merge['RMAG']                = 22.5 - 2.5 * np.log10(bright_merge['FLUX_R'])
bright_merge['RMAG_DRED']           = 22.5 - 2.5 * np.log10(bright_merge['FLUX_R'] / bright_merge['MW_TRANSMISSION_R'])

bright_merge['FIBER_RMAG']          = 22.5 - 2.5 * np.log10(bright_merge['FIBERFLUX_R'])
bright_merge['FIBER_RMAG_DRED']     = 22.5 - 2.5 * np.log10(bright_merge['FIBERFLUX_R'] / bright_merge['MW_TRANSMISSION_R'])

bright_merge['GMR']                 = bright_merge['GMAG'] - bright_merge['RMAG']
bright_merge['GMR_DRED']            = bright_merge['GMAG_DRED'] - bright_merge['RMAG_DRED']

# Target ever assigned (to a working fiber); deprecate: (bright_merge['ZWARN'].data != 999999) cut.   
# 1,108,673 reachable but fiberstatus == 999999 for unassigned.
#   241,429 with fiberstatus == 0
#    18,379 otherwise (fiberstatus > 0).
bright_merge['BGS_A_SUCCESS']       = (bright_merge['FIBERSTATUS'] == 0)

# Propagate to the other reachable instances for a target that was ever assigned to a working fiber. 
assigned = bright_merge['BGS_A_SUCCESS'] > 0
aids     = bright_merge['TARGETID'][assigned]

bright_merge['BGS_A_SUCCESS'][np.isin(bright_merge['TARGETID'], aids)] = 1.0
bright_merge.pprint()

if todisk:
    # List of reachable BGS Bright. 
    bright_merge.write('{}bright_reachable_sv3_v{:.1f}.fits'.format(odir, version), format='fits', overwrite=True)

bright_merge['BGS_A_WEIGHT']          = 1. / asuccess(bright_merge['RMAG_DRED'])

# Note: cannot use A_SUCCESS here, as must be on a working fiber, not at least once available to a working fiber. 
bright_merge_obs                      = bright_merge[(bright_merge['FIBERSTATUS'] == 0)]

# NOBS    TARGET_STATE
# 0       (array([b'BGS|UNOBS'],                         dtype='|S30'), array([    154424]))
# 1       (array([b'BGS|MORE_ZGOOD', b'BGS|MORE_ZWARN'], dtype='|S30'), array([ 149, 2338]))
# 2       (array([b'BGS|MORE_ZGOOD', b'BGS|MORE_ZWARN'], dtype='|S30'), array([   4, 1392]))
# 3       (array([b'BGS|DONE'],                          dtype='|S30'), array([         4]))

bright_merge_obs['BGS_Z_SUCCESS']     = (bright_merge_obs['ZWARN'] == 0) & (bright_merge_obs['DELTACHI2'] > 40.)
# bright_merge_obs['BGS_Z_SUCCESS']  &= (bright_merge_obs['ZTILEID'] > -1)
bright_merge_obs['BGS_Z_WEIGHT']      = 1. / zsuccess(bright_merge_obs['FIBER_RMAG'])

bright_merge_obs.pprint()

if todisk:
    bright_merge_obs.write('{}bright_sv3_v{:.1f}.fits'.format(odir, version), format='fits', overwrite=True)

exit(0)
    
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

    psys     = row['PHOTSYS']
    
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

            maxz    = zmax(kcorrector, rlim(psys), Mrh, gmr, zz, ref_gmr=ref_gmr)

            maxz    = np.minimum(maxz, params['zmax'])
            
            maxv    = vmax(kcorrector, rlim(psys), Mrh, gmr, zz, min_z=zmin(params['vmin']), fsky=sv3_params['fsky'], max_z=maxz)        
    
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

fails = np.array(fails)

if todisk:
    np.savetxt('{}bright_sv3_derivedfails_v{:.1f}.txt'.format(odir, version), fails, fmt='%d')
        
derived = Table(np.array(derived), names=['TARGETID', 'ROSETTE', 'BGS_Z_SUCCESS', 'INSAMPLE', 'BGS_Z_WEIGHT', 'BGS_A_WEIGHT', 'VOLUME', 'MRH', 'REF_GMR0P1', 'REF_GMR0P0', 'ZMAX', 'IVMAX', 'VONVMAX'])
derived['TARGETID'] = np.array(tids, dtype=np.int64)

print(np.mean(derived['INSAMPLE']))

derived.pprint(max_width=-1)

if todisk:
    derived.write('{}bright_sv3_derived_v{:.1f}.fits'.format(odir, version), format='fits', overwrite=True)

print('Finished writing to' + ' {}bright_sv3_derived_v{:.1f}.fits'.format(odir, version))
