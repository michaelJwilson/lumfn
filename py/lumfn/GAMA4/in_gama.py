import numpy as np

def in_gama(ras, decs):
    # https://www.astro.ljmu.ac.uk/~ikb/research/gama_fields/

    # ---- G02 ----
    isin = (ras > 30.2) & (ras < 38.8) & (decs > -10.25) & (decs < -3.72)

    # ---- G09 ----
    isin |= (ras > 129.) & (ras < 141.) & (decs > -2.00) & (decs < 3.00)

    # ---- G12 ----  
    isin |= (ras > 174.) & (ras < 186.) & (decs > -3.00) & (decs < 2.00)

    # ---- G15 ----
    isin |= (ras > 211.5) & (ras < 223.5) & (decs > -2.00) & (decs < 3.00)

    # TODO: G23

    return isin
