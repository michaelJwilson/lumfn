from params import params

def rlim(photsys, debug=True):
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
