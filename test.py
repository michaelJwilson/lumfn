import numpy as np
import pylab as pl

from   lumfn.tmr_kcorr import tmr_kcorr


x  = tmr_kcorr()
zs = np.arange(0.01, 0.51, 0.01)

gmrs = [0.158, 0.298, 0.419, 0.553, 0.708, 0.796, 0.960]
gmrs = [0.158, 0.960]

for gmr in [0.158, 0.298, 0.419, 0.553, 0.708, 0.796, 0.960]:
    ys = x.eval(gmr, zs)

    pl.plot(zs, ys, label=gmr)

pl.xlim(0.0, 0.5)
pl.ylim(-0.2, 1.2)
pl.xlabel('redshift')
pl.ylabel('$r$ $k$-correction')
pl.legend()
pl.show()
