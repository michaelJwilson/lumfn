import numpy             as     np
import pylab             as     pl
import astropy.io.fits   as     fits

from   astropy.table     import Table, unique
from   matplotlib.pyplot import figure
from   scipy.optimize    import curve_fit
from   matplotlib.pyplot import figure
from   zsuccess          import zsuccess
from   desitarget.sv3    import sv3_targetmask
from   scipy.stats       import linregress

figure(figsize=(6, 4), dpi=200)

# left merge of spec. to reachable targets.
bright   = Table.read('/global/cscratch1/sd/mjwilson/desi/BGS/lumfn/bright_reachable_sv3_v0.0.fits')
bright   = bright['TARGETID', 'RMAG_DRED', 'BGS_A_SUCCESS']

# One instance per target.  Was it ever assigned encoded by BGS_A_SUCCESS. 
bright   = unique(bright, keys='TARGETID', keep='first')

bright['RMAG_IDX'] = np.digitize(bright['RMAG_DRED'], bins=np.arange(12.0, 20.0, 0.1))

bright_grouped = bright.group_by(['RMAG_IDX'])
bright_binned  = bright_grouped['RMAG_DRED', 'RMAG_IDX', 'BGS_A_SUCCESS'].groups.aggregate(np.mean)

xs = np.arange(12.0, 21.0, 0.05)
pl.plot(bright_binned['RMAG_DRED'], bright_binned['BGS_A_SUCCESS'], c='k', lw=0.25, label='BGS BRIGHT', marker='.')

##  
asym = np.median(bright_binned['BGS_A_SUCCESS'])
pl.axhline(asym, lw=0.1)

##
##  slope, intercept, r, p, se = linregress(bright_binned['RMAG_DRED'], bright_binned['BGS_A_SUCCESS'])
##  pl.plot(xs, slope * xs + intercept, c='magenta', lw=0.25)

pl.xlabel('RMAG DRED')
pl.ylabel('BGS $A$ Success')

pl.legend(frameon=False, loc=1)

pl.ylim(-0.05, 1.05)

# pl.title('y={:.3f} (y={:.3f}x + {:.3f})'.format(asym, slope, intercept))
pl.title('y={:.3f}'.format(asym))

pl.show()
