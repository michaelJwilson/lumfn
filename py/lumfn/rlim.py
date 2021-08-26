from params import params

def rlim(photsys, debug=True):
    # https://github.com/desihub/desitarget/blob/19031803b56f83898a99112c02e8e5d671069c28/py/desitarget/sv3/sv3_cuts.py#L1358
    nom    = params['rlim']
    offset = 0.04

    if photsys in ['N', b'N']:
        return nom + offset
    
    elif photsys in ['S', b'S']:
        return nom
    
    else:
        if debug:
            print('WARNING: Photsys of {} found.  Undefined behaviour'.format(photsys))

        return nom
