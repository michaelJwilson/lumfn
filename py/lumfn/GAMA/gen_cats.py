import os
import sys
import time
import fitsio
import pylab              as     pl 
import numpy              as     np
import astropy.io.fits    as     fits

from   astropy.table      import Table, join, vstack, unique, hstack
from   desitarget.sv3     import sv3_targetmask
from   matplotlib.pyplot  import figure
from   scipy.optimize     import curve_fit

sys.path.append('/global/homes/m/mjwilson/desi/BGS/lumfn/bin')
sys.path.append('/global/homes/m/mjwilson/desi/BGS/lumfn/py/')
sys.path.append('/global/homes/m/mjwilson/desi/BGS/lumfn/py/lumfn/')

sys.path.append('/global/homes/m/mjwilson/desi/LSS/bin/')
sys.path.append('/global/homes/m/mjwilson/desi/LSS/py/')

from   distances          import comoving_distance, comoving_volume
from   ajs_kcorr          import ajs_kcorr
from   abs_mag            import abs_mag
from   vmax               import zmax, vmax
from   ref_gmr            import one_reference_gmr
from   LSS.SV3.cattools   import tile2rosette  
from   params             import params
from   zmin               import zmin
from   rlim               import rlim
from   desitarget.cuts    import notinBGS_mask
from   GAMA.define_sample import define_sample
from   GAMA.gama_params   import gama_params


version          = 0.2
todisk           = True
dryrun           = True
runtime_lim      = 0.1 # only if dryrun. 
odir             = os.environ['CSCRATCH'] + '/desi/BGS/lumfn/'

print(version, odir)

gama             = Table.read('/global/cfs/cdirs/desi/target/analysis/truth/dr9.0/south/matched/GAMA-DR3-SpecObj-match.fits')
gama['ZGAMA']    = gama['Z']

del gama['Z']

igama               = Table.read('/global/cfs/cdirs/desi/target/analysis/truth/dr9.0/south/matched/ls-dr9.0-GAMA-DR3-SpecObj-match.fits')

del igama['RA']
del igama['DEC']

bright_merge        = hstack([gama, igama])
bright_merge.pprint()

# ---- Apply BGS bright mask.
# https://github.com/desihub/desitarget/blob/19031803b56f83898a99112c02e8e5d671069c28/py/desitarget/cuts.py#L1426

not_masked = notinBGS_mask(gnobs=bright_merge['NOBS_G'], rnobs=bright_merge['NOBS_R'], znobs=bright_merge['NOBS_Z'], primary=None,
                                          gfluxivar=bright_merge['FLUX_IVAR_G'], rfluxivar=bright_merge['FLUX_IVAR_R'], zfluxivar=bright_merge['FLUX_IVAR_Z'], Grr=0.65*np.ones_like(bright_merge['NOBS_G']),
                                          gaiagmag=None, maskbits=bright_merge['MASKBITS'], targtype='bright')
bright_merge = bright_merge[not_masked]

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
bright_merge['BGS_A_SUCCESS']       = True

if todisk:
    # List of reachable BGS Bright. 
    bright_merge.write('{}/GAMA/bright_reachable_v{:.1f}.fits'.format(odir, version), format='fits', overwrite=True)
    
bright_merge['BGS_A_WEIGHT']          = 1.
bright_merge_obs                      = bright_merge

bright_merge_obs['BGS_Z_SUCCESS']     = True
bright_merge_obs['BGS_Z_WEIGHT']      = 1.

bright_merge_obs.pprint()

if todisk:
    bright_merge_obs.write('{}/GAMA/bright_v{:.1f}.fits'.format(odir, version), format='fits', overwrite=True)

## 
derived      = []

start        = time.time()

kcorrector   = ajs_kcorr()

fails        = []
tids         = []

for ii, row in enumerate(bright_merge_obs):
    tid      = ii
    tids.append(tid)
    
    ros      = -99.
    
    rmag     = row['RMAG_DRED']
    gmr      = row['GMR_DRED']
    zz       = row['ZGAMA']

    psys     = 'S'
    
    #  See dedicated fsky calc. notebook. 
    vol      = comoving_volume(zmin(params['vmin']), zz, fsky=gama_params['fsky'])
    
    zwght    = row['BGS_Z_WEIGHT']
    awght    = row['BGS_A_WEIGHT']

    zsuccess = row['BGS_Z_SUCCESS']

    insample = define_sample(row)
    
    void     = [tid, ros, zsuccess, False, zwght, awght, -99., -99., -99., -99., -99., -99., -99.]

    if zsuccess & insample:
        try:            
            Mrh     = abs_mag(kcorrector, rmag, gmr, zz).item()

            org_gmr = one_reference_gmr(kcorrector, gmr, zz, zref=kcorrector.z0, ecorr=False)
            
            ref_gmr = one_reference_gmr(kcorrector, gmr, zz, zref=params['ref_z'])[0]

            maxz    = zmax(kcorrector, rlim(psys), Mrh, gmr, zz, ref_gmr=ref_gmr)

            maxz    = np.minimum(maxz, params['zmax'])
            
            maxv    = vmax(kcorrector, rlim(psys), Mrh, gmr, zz, min_z=zmin(params['vmin']), fsky=gama_params['fsky'], max_z=maxz)        
    
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
    np.savetxt('{}/GAMA/bright_derivedfails_v{:.1f}.txt'.format(odir, version), fails, fmt='%d')
        
derived = Table(np.array(derived), names=['TARGETID', 'ROSETTE', 'BGS_Z_SUCCESS', 'INSAMPLE', 'BGS_Z_WEIGHT', 'BGS_A_WEIGHT', 'VOLUME', 'MRH', 'REF_GMR0P1', 'REF_GMR0P0', 'ZMAX', 'IVMAX', 'VONVMAX'])
derived['TARGETID'] = np.array(tids, dtype=np.int64)

print(np.mean(derived['INSAMPLE']))

derived.pprint(max_width=-1)

if todisk:
    derived.write('{}/GAMA/bright_derived_v{:.1f}.fits'.format(odir, version), format='fits', overwrite=True)

print('Finished writing to' + ' {}/GAMA/bright_derived_v{:.1f}.fits'.format(odir, version))
