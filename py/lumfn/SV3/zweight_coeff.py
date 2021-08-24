import numpy             as     np
import pylab             as     pl
import astropy.io.fits   as     fits

from   astropy.table     import Table
from   matplotlib.pyplot import figure
from   scipy.optimize    import curve_fit
from   matplotlib.pyplot import figure
from   zsuccess          import zsuccess


figure(figsize=(6, 4), dpi=200)

bright = Table.read('/global/cscratch1/sd/mjwilson/desi/BGS/lumfn/bright_sv3_v0.0.fits')

print(bright.dtype.names)

bright['RMAG_IDX'] = np.digitize(bright['FIBER_RMAG'], bins=np.arange(19.5, 22., 0.2))

bright_grouped = bright.group_by(['RMAG_IDX'])
bright_binned  = bright_grouped['TSNR2_BGS', 'FIBER_RMAG', 'RMAG', 'RMAG_DRED', 'RMAG_IDX', 'BGS_Z_SUCCESS'].groups.aggregate(np.mean)

bright_binned.sort('FIBER_RMAG')

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
popt, pcov = curve_fit(zsuccess, bright_binned['FIBER_RMAG'], bright_binned['BGS_Z_SUCCESS'], p0=[1.31, 22.3])

print(popt)

## 
xs = np.arange(19.0, 23.0, 0.05)

pl.plot(bright_binned['FIBER_RMAG'], bright_binned['BGS_Z_SUCCESS'], c='k', lw=0.25, label='BGS BRIGHT')
pl.plot(xs, zsuccess(xs, a=popt[0], b=popt[1]), label='Best fit', alpha=0.5)
pl.plot(xs, zsuccess(xs, a=1.31,    b=22.314),  label='Sam',      alpha=0.5)

pl.xlabel('FIBER_RMAG')
pl.ylabel('BGS $z$ Success')

pl.legend(frameon=False, loc=1)

pl.ylim(-0.05, 1.05)

pl.title('({:.3f},{:.3f})'.format(popt[0], popt[1]))

pl.show()
