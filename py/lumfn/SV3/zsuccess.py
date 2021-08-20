import numpy as np


def zsuccess(rfiber, a=2.68, b=22.113):
    # v0: [2.68, 22.113].                                                                                                                                                                                                
    return 1. /  (1. + np.exp(a * (rfiber - b)))
