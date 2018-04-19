import numpy as np
def uniform_scaling(Q,p):
    Qp=np.empty(p,dtype='object')
    n=len(Q)
    for j in range(p):
        Qp[j]=Q[int(np.floor(j*n/p))]
    return Qp
