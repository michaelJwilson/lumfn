import numpy as np

def zsuccess(rfiber, a=2.371, b=22.406):
    # v0: [2.68, 22.113].                                                                                                                                                                                                                    
    # v1: [2.37140526, 22.4056713] - inclusion of stellar zlo and zhi=0.5 in cat.                                                                                                                                                             
    return 1. /  (1. + np.exp(a * (rfiber - b)))
